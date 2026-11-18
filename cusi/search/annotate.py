"""
Annotate a finished search run with the tasks actually done along its paths, written in the form of the
benchmark's train tasks (plans/annotation_plan.md). One run does both stages and continues where it stopped:

1. Transition reports: every edge (a node's segment from its parent) without a `description` gets one from
   TransitionDescriber (its parent's screen, then every step to it), as a VLM-explorer search writes them as it goes;
   random and curiosity trees run without a model, so this gives them the same reports. Written to nodes.jsonl as
   update rows ("description", "window_descriptions").
2. Path tasks: every node starts a chain on every branch below it. A chain is extended one edge at a time
   (cusi.search.path_notes): while it has no task, a call says whether the chain from its start now completes one
   (TOO_SHORT keeps going); once it has one, each further edge is CONTINUE (no effect on the task: the chain goes on,
   the task's end stays), REVISE (a larger or more specific task now completed; the earlier name stays as a task of
   its own, ending where it ended) or NEW (the task ended before this edge; the chain stops). A chain is copied at
   every fork, and stops at a leaf, a challenge page, or when it would pass --max_steps env steps. A chain with no
   task by then keeps the best partial task its TOO_SHORT replies named (the latest; "partial": true), or is dropped
   if they named none. Every call sees a fixed sample of the benchmark's train tasks (cusi.search.task_context).

Each call's verdict is cached in annotation/p<prompt version>/calls.jsonl (keyed by the chain's node path), so a
rerun continues where it stopped and rebuilds the same tasks. Results, in the run directory, annotation/p<prompt
version>/:
- tasks.jsonl: one row per task: {"start", "task", "end", "path" (node ids start..end), "n_edges", "n_steps",
  "revisions" (the chain's names in order, up to this one), "nested" (an earlier name that a REVISE replaced),
  "partial" (a chain's best partial task, kept because it never named a full one)};
- meta.json: model, settings, the train sample (ids) and the split file's hash; refuses mixed settings;
- summary.json: counts, written last (a run without it is unfinished).

    python -m cusi.search.annotate --env android --run sat_random_s0 --model_name <model> --vllm_base_url <url>
"""
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Any, Optional
import click
from cusi.agents.frames import FrameRef
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.search import path_notes as pn
from cusi.search.store import EXPANSIONS_FILE, append_node_updates, node_rows, read_rows, step_from_row
from cusi.search.task_context import TrainContext, tree_group
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run

ANNOTATION_DIR = "annotation"
CALLS_FILE, TASKS_FILE, META_FILE, SUMMARY_FILE = "calls.jsonl", "tasks.jsonl", "meta.json", "summary.json"
MAX_NEW_TOKENS = 300
#: The transition-report fields stage 1 writes (TransitionDescriber's output; "discovered" stays the search's).
REPORT_FIELDS = ("description", "window_descriptions")


def _same(*, a: str, b: str) -> bool:
    """Two task names that differ only in case, spacing and punctuation."""
    def norm(s):
        return " ".join("".join(c for c in str(s).lower() if c.isalnum() or c.isspace()).split())
    return norm(a) == norm(b)


def annotation_dir(*, directory: str) -> str:
    """annotation/p<prompt version>/: each prompt version's verdicts and tasks apart."""
    return os.path.join(directory, ANNOTATION_DIR, f"p{pn.PROMPT_VERSION}")


def read_tasks(*, directory: str) -> list:
    """The run's path tasks (tasks.jsonl rows)."""
    path = os.path.join(annotation_dir(directory=directory), TASKS_FILE)
    if not os.path.exists(path):
        raise click.UsageError(f"{directory} has no tasks: run cusi.search.annotate first")
    return read_rows(path)


def unfinished(*, directory: str) -> bool:
    """An annotation was started on the run (its meta.json exists) and has not written its summary since."""
    d = annotation_dir(directory=directory)
    meta, summary = os.path.join(d, META_FILE), os.path.join(d, SUMMARY_FILE)
    return os.path.exists(meta) and (not os.path.exists(summary) or os.path.getmtime(meta) > os.path.getmtime(summary))


def node_app(*, env_name: str, node: dict, expansions: dict):
    """Android: the app on a node's screen, named so its tasks name it right (the annotator, seeing only the screen,
    called other apps' settings "the Settings app" and many apps "the app"): the foreground package recorded on the
    step that reached the node (the env's true state), else, for a root, its scene's app. None elsewhere."""
    if env_name != "android":
        return None
    from cusi.agents.specs import android_app_label
    if node.get("segment"):
        exp_id, _, last = node["segment"]
        true = ((expansions[exp_id]["steps"][last].get("info") or {}).get("true") or {})
        if true.get("activity"):
            return android_app_label(package=true["activity"].split("/")[0])
    scene = ENV_SPECS["android"].scenes.get(node.get("env_key"))
    return scene.kwargs.get("app_label") if scene else None


def env_description(*, env_name: str, directory: str) -> str:
    """The env as the prompts name it: GameBoy names the tree's own game."""
    if env_name != "gameboy":
        return ENV_SPECS[env_name].env_description
    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)
    return f"the GameBoy game {tree_group(env=env_name, config=config.get('config', config))}"


def describe_edges(*, directory: str, env_name: str, vlm, workers: int, parameters: dict[str, Any]) -> int:
    """Stage 1: a transition report for every edge that has none (challenge pages skipped). Returns how many."""
    from cusi.search.describe import TransitionDescriber
    describer = TransitionDescriber(vlm=vlm, parameters=parameters)
    description = env_description(env_name=env_name, directory=directory)
    nodes = node_rows(directory=directory)
    expansions = {e["id"]: e for e in read_rows(os.path.join(directory, EXPANSIONS_FILE))}
    todo = [nid for nid in sorted(nodes) if nodes[nid]["parent"] is not None and nodes[nid].get("description") is None
            and not (nodes[nid].get("flags") or {}).get("challenge_page")]
    log_info(f"annotate {directory}: transition reports for {len(todo)} edges", parameters=parameters)

    def report(nid: int) -> dict:
        exp_id, first, last = nodes[nid]["segment"]
        steps = [step_from_row(directory=directory, row=r) for r in expansions[exp_id]["steps"][first:last + 1]]
        start = FrameRef(path=os.path.join(directory, nodes[nodes[nid]["parent"]]["frame"]))
        out = describer(env_description=description, steps=steps, start_frame=start) or {}
        return {k: out.get(k) for k in REPORT_FIELDS}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(report, nid): nid for nid in todo}
        for future in as_completed(futures):
            append_node_updates(directory=directory, rows=[{"id": futures[future], **future.result()}])
            done += 1
            if done % 100 == 0:
                log_info(f"annotate: {done}/{len(todo)} transition reports", parameters=parameters)
    return done


class PathAnnotator:
    def __init__(self, *, directory: str, env_name: str, vlm, model_name: str, max_steps: int, n_per_flavour: int,
                 sample_seed: int, workers: int, max_starts: Optional[int], parameters: dict[str, Any]) -> None:
        """Stage 2 (stage 1 must have run). max_starts: annotate only chains from a seeded sample of this many start
        nodes (a pilot); None: all."""
        self._parameters = parameters
        self.directory, self.env_name, self.vlm = directory, env_name, vlm
        self.max_steps, self.workers = max_steps, workers
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        group = tree_group(env=env_name, config=config.get("config", config))
        self.context = TrainContext(env=env_name, group=group, n_per_flavour=n_per_flavour, seed=sample_seed,
                                    parameters=parameters)
        self.train_block = self.context.block()
        self.env_description = env_description(env_name=env_name, directory=directory)
        self.nodes = node_rows(directory=directory)
        self.expansions = {e["id"]: e for e in read_rows(os.path.join(directory, EXPANSIONS_FILE))}
        self.kids = {}
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            if n["parent"] is not None and not (n.get("flags") or {}).get("challenge_page"):
                self.kids.setdefault(n["parent"], []).append(nid)
        starts = [nid for nid in sorted(self.nodes) if self.kids.get(nid)
                  and not (self.nodes[nid].get("flags") or {}).get("challenge_page")]
        if max_starts is not None and max_starts < len(starts):
            starts = sorted(random.Random(f"starts:{sample_seed}").sample(starts, max_starts))
        self.starts = starts
        self.out_dir = annotation_dir(directory=directory)
        os.makedirs(self.out_dir, exist_ok=True)
        meta = {"model_name": model_name, "max_steps": max_steps, "prompt_version": pn.PROMPT_VERSION,
                **self.context.meta}
        meta_path = os.path.join(self.out_dir, META_FILE)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                done = json.load(f)
            if {k: done.get(k) for k in meta} != meta:
                raise click.UsageError(f"{directory} was annotated with {done}; not mixing in {meta}")
        with open(meta_path, "w") as f:
            json.dump({**meta, "max_starts": max_starts}, f, indent=1)
        self.calls_path = os.path.join(self.out_dir, CALLS_FILE)
        self.cache = {r["key"]: r for r in read_rows(self.calls_path)}
        self._lock = threading.Lock()
        self.counts = {"calls": 0, "cached": 0, "unparsed": 0, "dropped_building": 0, "kept_partial": 0,
                       "capped": 0}

    # ------------------------------------------------------------------ one edge
    def edge_steps(self, *, nid: int) -> list:
        exp_id, first, last = self.nodes[nid]["segment"]
        return self.expansions[exp_id]["steps"][first:last + 1]

    def frame(self, *, nid: int):
        return FrameRef(path=os.path.join(self.directory, self.nodes[nid]["frame"])).pil()

    def ask(self, *, chain: dict, child: int) -> dict:
        """The cached or new verdict on extending chain by the edge to child."""
        key = ":".join(str(x) for x in chain["path"] + [child])
        with self._lock:
            hit = self.cache.get(key)
        if hit is not None:
            with self._lock:
                self.counts["cached"] += 1
            return hit
        steps = [step_from_row(directory=self.directory, row=r) for r in self.edge_steps(nid=child)]
        reports = [self.nodes[n].get("description") for n in chain["path"][1:]]
        app = node_app(env_name=self.env_name, node=self.nodes[child], expansions=self.expansions)
        args = dict(env_description=self.env_description, app_note=pn.app_note(app=app, where="Image 3"),
                    reports=reports, steps=steps, latest=self.nodes[child].get("description"),
                    train=self.train_block)
        if chain["task"] is None:
            prompt, allowed = pn.build_prompt(**args, env_name=self.env_name), (pn.TOO_SHORT, pn.TASK)
        else:
            prompt, allowed = pn.extend_prompt(**args, current=chain["task"]), (pn.CONTINUE, pn.REVISE, pn.NEW)
        images = [self.frame(nid=chain["path"][0]), self.frame(nid=chain["path"][-1]), self.frame(nid=child)]
        out = str(self.vlm.infer(texts=prompt, images=images, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=0.0)["output"])
        verdict, task, parsed = pn.parse_verdict(text=out, allowed=allowed)
        partial = pn.parse_partial(text=out) if verdict == pn.TOO_SHORT else None
        row = {"key": key, "verdict": verdict, "task": task, "partial": partial, "parsed": parsed, "raw": out}
        with self._lock:
            self.counts["calls"] += 1
            self.counts["unparsed"] += not parsed
            self.cache[key] = row
            with open(self.calls_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    # ------------------------------------------------------------------ chains
    def advance(self, *, chain: dict, child: int) -> tuple:
        """(chains to extend further as (chain, child) pairs, finished chains) after the edge to child."""
        n_steps = len(self.edge_steps(nid=child))
        if chain["n_steps"] + n_steps > self.max_steps:
            with self._lock:
                self.counts["capped"] += 1
            return [], [chain]
        row = self.ask(chain=chain, child=child)
        verdict = row["verdict"]
        if not row["parsed"]:          # unreadable: no task yet keeps building, a named task is ended (conservative)
            verdict = pn.TOO_SHORT if chain["task"] is None else pn.NEW
        if verdict == pn.NEW:
            return [], [chain]
        nxt = {**chain, "path": chain["path"] + [child], "n_steps": chain["n_steps"] + n_steps,
               "names": list(chain["names"])}
        if verdict == pn.TOO_SHORT and row.get("partial"):
            # the best task the chain has completed so far: kept if the chain never names a full one
            nxt["partial"] = (row["partial"], child)
        if verdict == pn.TASK:
            nxt["task"], nxt["names"] = row["task"], [(row["task"], child)]
        elif verdict == pn.REVISE and _same(a=row["task"], b=chain["task"]):
            # the same name again: the task was achieved further on; its end moves, no nested copy
            nxt["names"][-1] = (nxt["names"][-1][0], child)
        elif verdict == pn.REVISE:
            nxt["task"] = row["task"]
            nxt["names"].append((row["task"], child))
        # CONTINUE (no effect on the task): the chain goes on, the task's end stays where it was achieved
        kids = self.kids.get(child) or []
        if not kids:
            return [], [nxt]
        return [(nxt, k) for k in kids], []

    def task_row(self, *, chain: dict, name: str, end: int, revisions: list, nested: bool, partial: bool) -> dict:
        path = chain["path"][:chain["path"].index(end) + 1]
        return {"start": chain["path"][0], "task": name, "end": end, "path": path, "n_edges": len(path) - 1,
                "n_steps": sum(len(self.edge_steps(nid=n)) for n in path[1:]), "revisions": revisions,
                "nested": nested, "partial": partial}

    def tasks_of(self, *, chain: dict) -> list:
        """The finished chain's task rows: its last name (to its end) and every name a REVISE replaced (nested, to
        where that name ended); a chain without one, its best partial task, if any."""
        if chain["task"] is None:
            if not chain.get("partial"):
                return []
            name, end = chain["partial"]
            return [self.task_row(chain=chain, name=name, end=end, revisions=[name], nested=False, partial=True)]
        names = chain["names"]
        return [self.task_row(chain=chain, name=name, end=end, revisions=[x[0] for x in names[:i + 1]],
                              nested=i < len(names) - 1, partial=False) for i, (name, end) in enumerate(names)]

    def run(self) -> dict:
        t0 = time.time()
        log_info(f"annotate {self.directory}: {len(self.starts)} start nodes, {len(self.cache)} cached calls, "
                 f"{self.workers} at a time", parameters=self._parameters)
        pending = [({"path": [u], "task": None, "names": [], "n_steps": 0}, k) for u in self.starts
                   for k in self.kids[u]]
        finished, done_edges = [], 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            running = set()
            while pending or running:
                while pending and len(running) < 4 * self.workers:
                    chain, child = pending.pop()
                    running.add(pool.submit(self.advance, chain=chain, child=child))
                ready, running = wait(running, return_when=FIRST_COMPLETED)
                for fut in ready:
                    more, ended = fut.result()
                    pending += more
                    finished += ended
                    done_edges += 1
                    if done_edges % 200 == 0:
                        log_info(f"annotate: {done_edges} edges, {len(pending) + len(running)} open, "
                                 f"{self.counts}", parameters=self._parameters)
        rows, seen = [], set()
        for chain in finished:
            if chain["task"] is None:
                self.counts["kept_partial" if chain.get("partial") else "dropped_building"] += 1
            for r in self.tasks_of(chain=chain):
                key = (r["start"], r["end"], r["task"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
        rows.sort(key=lambda r: (r["start"], r["end"], r["task"]))
        with open(os.path.join(self.out_dir, TASKS_FILE), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary = {**self.counts, "tasks": len(rows), "distinct_texts": len({r["task"] for r in rows}),
                   "starts": len(self.starts), "starts_with_task": len({r["start"] for r in rows}),
                   "nodes": len(self.nodes), "seconds": round(time.time() - t0)}
        with open(os.path.join(self.out_dir, SUMMARY_FILE), "w") as f:
            json.dump(summary, f, indent=1)
        return summary


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "run_name", required=True)
@click.option("--model_name", required=True)
@click.option("--model_backend", default="vllm", show_default=True)
@click.option("--vllm_base_url", default=None)
@click.option("--max_steps", default=30, show_default=True, help="Env steps a chain may span.")
@click.option("--n_per_flavour", default=3, show_default=True, help="Train tasks shown per flavour.")
@click.option("--sample_seed", default=0, show_default=True, help="Seed of the train sample (and of --max_starts).")
@click.option("--max_starts", default=None, type=int, help="A pilot: only this many start nodes (seeded sample).")
@click.option("--workers", default=16, show_default=True, help="Calls at a time.")
def main(env_name, run_name, model_name, model_backend, vllm_base_url, max_steps, n_per_flavour, sample_seed,
         max_starts, workers):
    from cusi.agents.vlm import AgentVLM
    parameters = load_parameters()
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    vlm = AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                   parameters=parameters)
    n = describe_edges(directory=directory, env_name=env_name, vlm=vlm, workers=workers, parameters=parameters)
    summary = PathAnnotator(directory=directory, env_name=env_name, vlm=vlm, model_name=model_name,
                            max_steps=max_steps, n_per_flavour=n_per_flavour, sample_seed=sample_seed,
                            workers=workers, max_starts=max_starts, parameters=parameters).run()
    log_info(f"annotate {directory}: {n} transition reports written; {summary}", parameters=parameters)


if __name__ == "__main__":
    main()
