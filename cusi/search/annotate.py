"""
Annotate a finished search run with the tasks actually done along its paths, written in the form of the
benchmark's train tasks (plans/annotation_plan.md). One run, by one annotation model, does three stages and continues
where it stopped:

1. Transition reports: a VLM-explorer search writes a report of every edge it makes (its base rows' `description`
   and `window_descriptions`), and those are used as they are. Every other edge (random and curiosity trees run
   without a model, and any gap) gets one from the annotation model's TransitionDescriber (its parent's screen, then
   every step to it), written to the model's transitions.jsonl; nodes.jsonl is never written.
2. Path tasks: every node starts a chain on every branch below it. A chain is extended one edge at a time
   (cusi.search.path_notes): while it has no task, a call says whether the chain from its start now completes one
   (TOO_SHORT keeps going); once it has one, each further edge is CONTINUE (no effect on the task: the chain goes on,
   the task's end stays), REVISE (a larger or more specific task now completed; the earlier name stays as a task of
   its own, ending where it ended) or NEW (the task ended before this edge; the chain stops). A chain is copied at
   every fork, and stops at a leaf, a challenge page, or when it would pass --max_steps env steps. A chain with no
   task by then keeps the best partial task its TOO_SHORT replies named (the latest; "partial": true), or is dropped
   if they named none. Every call sees a fixed sample of the benchmark's train tasks (cusi.search.task_context).
3. Fit gate: one call per candidate task (its start screen, the path's reports, its end screen, the train sample)
   accepts it only if the path completes it, the end screen shows it done or shows the answer, it reads like a train
   task, and it is not a one-click wrapper; verdicts cached in gate_calls.jsonl.

In Web and Android (path_notes.STEP_ENVS), whose edges run 5-19 steps, stage 2 also reads the latest edge's
screen-by-screen report and names the action after which a task was complete: a task ends at (edge end node, step)
and records how to replay it (the edge's start node and its first actions); the gate sees that edge's screens up to
that action; a task needs a checkable outcome (no "find information about X", no navigation tacked onto a done task,
no exploration-label search queries); accepted paraphrases of one start are merged (alt_ends).

Results, in the run directory, annotation/<annotation model's short name>/ (models side by side):
- transitions.jsonl: stage 1's reports ({"id", "description", "window_descriptions"});
- calls.jsonl: stage 2's verdicts, keyed by the chain's node path, so a rerun continues where it stopped and rebuilds
  the same tasks; gate_calls.jsonl: the gate's;
- candidates.jsonl: one row per stage-2 task: {"start", "task", "end", "end_step", "path" (node ids start..end),
  "n_edges", "n_steps", "mid_edge", "replay", "restorable", "revisions" (the chain's names in order, up to this one),
  "nested" (an earlier name that a REVISE replaced), "partial" (a chain's best partial task, kept because it never
  named a full one)};
- tasks.jsonl: the candidates the gate accepted (the annotation's tasks); rejected.jsonl: the others, with the
  gate's "failed" criteria and "reason";
- meta.json: model, settings, the train sample (ids), the split file's hash, how many reports came from the explorer
  and how many from the annotation model; refuses mixed settings;
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
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows, step_from_row
from cusi.search.task_context import TrainContext, tree_group
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run

ANNOTATION_DIR = "annotation"
CALLS_FILE, TASKS_FILE, META_FILE, SUMMARY_FILE = "calls.jsonl", "tasks.jsonl", "meta.json", "summary.json"
CANDIDATES_FILE, REJECTED_FILE, GATE_CALLS_FILE = "candidates.jsonl", "rejected.jsonl", "gate_calls.jsonl"
TRANSITIONS_FILE = "transitions.jsonl"
#: Runs never annotated (user's call): random-step fills of another tree (run_search --sweep_expansions), cuts of
#: another tree at a compute budget (cusi.search.cut), smoke and test runs.
NOT_ANNOTATED_SUFFIXES, NOT_ANNOTATED_PREFIXES = ("_random_filled", "_cut69h"), ("smoke_", "tt_")
MAX_NEW_TOKENS = 300
#: The transition-report fields stage 1 writes (TransitionDescriber's output; "discovered" stays the search's).
REPORT_FIELDS = ("description", "window_descriptions")
#: STEP_ENVS gate: at most this many screens of the last edge (its start, then each action's result up to the
#: completing one).
GATE_MAX_EDGE_SCREENS = 11
#: STEP_ENVS: accepted tasks of one start whose names share this much of their content words are one task.
PARAPHRASE_JACCARD = 0.8
_FILLER = {"the", "a", "an", "in", "on", "of", "to", "for", "app", "application", "and", "s"}


def annotated_run(*, run_name: str) -> bool:
    """Whether a run is one the annotation covers (not a fill, a cut, a smoke or a test run)."""
    return not (run_name.endswith(NOT_ANNOTATED_SUFFIXES) or run_name.startswith(NOT_ANNOTATED_PREFIXES))


def _same(*, a: str, b: str) -> bool:
    """Two task names that differ only in case, spacing and punctuation."""
    def norm(s):
        return " ".join("".join(c for c in str(s).lower() if c.isalnum() or c.isspace()).split())
    return norm(a) == norm(b)


def model_short_name(*, model_name: str) -> str:
    """A model's directory name: its name without the organisation (google/gemma-4-31b-it -> gemma-4-31b-it)."""
    return model_name.rstrip("/").split("/")[-1]


def annotation_dir(*, directory: str, model_name: str) -> str:
    """annotation/<model short name>/: one annotation model's reports, verdicts and tasks."""
    return os.path.join(directory, ANNOTATION_DIR, model_short_name(model_name=model_name))


def read_tasks(*, directory: str, model_name: str) -> list:
    """The run's tasks as annotated by model_name (tasks.jsonl rows)."""
    path = os.path.join(annotation_dir(directory=directory, model_name=model_name), TASKS_FILE)
    if not os.path.exists(path):
        raise click.UsageError(f"{directory} has no tasks by {model_name}: run cusi.search.annotate first")
    return read_rows(path)


def unfinished(*, directory: str) -> list:
    """The annotation models whose annotation of the run was started (meta.json) and has not written its summary
    since."""
    root, out = os.path.join(directory, ANNOTATION_DIR), []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        meta, summary = os.path.join(root, name, META_FILE), os.path.join(root, name, SUMMARY_FILE)
        if os.path.exists(meta) and (not os.path.exists(summary)
                                     or os.path.getmtime(meta) > os.path.getmtime(summary)):
            out.append(name)
    return out


def transitions_path(*, directory: str, model_name: str) -> str:
    return os.path.join(annotation_dir(directory=directory, model_name=model_name), TRANSITIONS_FILE)


def read_reports(*, directory: str, nodes: dict, model_name: str, field: str = "description") -> dict:
    """node id -> the report of the edge into it (field "description", or its screen-by-screen "window_descriptions"):
    the explorer's own (a VLM explorer's base row), else the annotation model's stage-1 report."""
    out = {nid: n[field] for nid, n in nodes.items() if n.get(field) is not None}
    for r in read_rows(transitions_path(directory=directory, model_name=model_name)):
        out.setdefault(r["id"], r.get(field))
    return out


def report_sources(*, directory: str, nodes: dict, model_name: str) -> dict:
    """How many edges' reports come from the explorer (base rows) and how many from the annotation model."""
    explorer = {nid for nid, n in nodes.items() if n["parent"] is not None and n.get("description") is not None}
    model = {r["id"] for r in read_rows(transitions_path(directory=directory, model_name=model_name))} - explorer
    return {"explorer": len(explorer), "annotation_model": len(model),
            "edges": sum(n["parent"] is not None for n in nodes.values())}


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


def children(*, nodes: dict) -> dict:
    """node id -> its children's ids, in id order; challenge pages are no node's child (chains never enter them)."""
    kids = {}
    for nid in sorted(nodes):
        n = nodes[nid]
        if n["parent"] is not None and not (n.get("flags") or {}).get("challenge_page"):
            kids.setdefault(n["parent"], []).append(nid)
    return kids


def start_nodes(*, nodes: dict, kids: dict, max_starts: Optional[int], seed: int) -> list:
    """Every node with a child (not a challenge page), or a seeded sample of max_starts of them."""
    starts = [nid for nid in sorted(nodes)
              if kids.get(nid) and not (nodes[nid].get("flags") or {}).get("challenge_page")]
    if max_starts is not None and max_starts < len(starts):
        starts = sorted(random.Random(f"starts:{seed}").sample(starts, max_starts))
    return starts


def reachable_edges(*, nodes: dict, kids: dict, starts: list, max_steps: int) -> set:
    """The child ids of the edges a chain from starts can extend into within max_steps env steps."""
    out = set()
    for u in starts:
        frontier = [(k, 0) for k in kids.get(u, [])]
        while frontier:
            nid, used = frontier.pop()
            _, first, last = nodes[nid]["segment"]
            used += last - first + 1
            if used > max_steps:
                continue
            out.add(nid)
            frontier += [(k, used) for k in kids.get(nid, [])]
    return out


def describe_edges(*, directory: str, env_name: str, vlm, model_name: str, workers: int, parameters: dict[str, Any],
                   only: Optional[set] = None) -> int:
    """Stage 1: a transition report for every edge that has none (challenge pages skipped), or only for the edges
    into the nodes in only (a sample of starts' reach). Returns how many."""
    from cusi.search.describe import TransitionDescriber
    describer = TransitionDescriber(vlm=vlm, parameters=parameters)
    description = env_description(env_name=env_name, directory=directory)
    nodes = node_rows(directory=directory)
    expansions = {e["id"]: e for e in read_rows(os.path.join(directory, EXPANSIONS_FILE))}
    reports = read_reports(directory=directory, nodes=nodes, model_name=model_name)
    todo = [nid for nid in sorted(nodes) if nodes[nid]["parent"] is not None and reports.get(nid) is None
            and not (nodes[nid].get("flags") or {}).get("challenge_page") and (only is None or nid in only)]
    path = transitions_path(directory=directory, model_name=model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
            with open(path, "a") as f:
                f.write(json.dumps({"id": futures[future], **future.result()}) + "\n")
            done += 1
            if done % 100 == 0:
                log_info(f"annotate: {done}/{len(todo)} transition reports", parameters=parameters)
    return done


def _content_words(text: str) -> set:
    words = "".join(c if c.isalnum() else " " for c in str(text).lower()).split()
    return {w for w in words if w not in _FILLER}


def merge_paraphrases(*, rows: list) -> list:
    """Accepted tasks of one start node whose names are paraphrases (content-word Jaccard >= PARAPHRASE_JACCARD) become
    one row: the shortest (fewest steps), with the others' names as "paraphrases" and their ends as "alt_ends"."""
    out = []
    by_start = {}
    for r in rows:
        by_start.setdefault(r["start"], []).append(r)
    for start in sorted(by_start):
        groups = []
        for r in sorted(by_start[start], key=lambda r: (r["n_steps"], r["task"])):
            words = _content_words(r["task"])
            for g in groups:
                lead = _content_words(g[0]["task"])
                if len(words & lead) / max(1, len(words | lead)) >= PARAPHRASE_JACCARD:
                    g.append(r)
                    break
            else:
                groups.append([r])
        for g in groups:
            out.append({**g[0], "paraphrases": sorted({x["task"] for x in g[1:]} - {g[0]["task"]}),
                        "alt_ends": [{"end": x["end"], "end_step": x["end_step"], "path": x["path"]} for x in g[1:]]})
    return out


class PathAnnotator:
    def __init__(self, *, directory: str, env_name: str, vlm, model_name: str, max_steps: int, n_per_flavour: int,
                 sample_seed: int, workers: int, max_starts: Optional[int], parameters: dict[str, Any]) -> None:
        """Stage 2 (stage 1 must have run). max_starts: annotate only chains from a seeded sample of this many start
        nodes; None: all."""
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
        self.reports = read_reports(directory=directory, nodes=self.nodes, model_name=model_name)
        self.windows = read_reports(directory=directory, nodes=self.nodes, model_name=model_name,
                                    field="window_descriptions")
        #: Web and Android: tasks end at the step that completes them, and accepted paraphrases are merged
        self.steps = env_name in pn.STEP_ENVS
        self.expansions = {e["id"]: e for e in read_rows(os.path.join(directory, EXPANSIONS_FILE))}
        self.kids = children(nodes=self.nodes)
        self.starts = start_nodes(nodes=self.nodes, kids=self.kids, max_starts=max_starts, seed=sample_seed)
        self.out_dir = annotation_dir(directory=directory, model_name=model_name)
        os.makedirs(self.out_dir, exist_ok=True)
        meta = {"model_name": model_name, "max_steps": max_steps, **self.context.meta}
        meta_path = os.path.join(self.out_dir, META_FILE)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                done = json.load(f)
            if {k: done.get(k) for k in meta} != meta:
                raise click.UsageError(f"{directory} was annotated with {done}; not mixing in {meta}")
        with open(meta_path, "w") as f:
            json.dump({**meta, "max_starts": max_starts,
                       "reports": report_sources(directory=directory, nodes=self.nodes, model_name=model_name)},
                      f, indent=1)
        self.calls_path = os.path.join(self.out_dir, CALLS_FILE)
        self.cache = {r["key"]: r for r in read_rows(self.calls_path)}
        self._lock = threading.Lock()
        self.counts = {"calls": 0, "cached": 0, "unparsed": 0, "dropped_building": 0, "kept_partial": 0,
                       "capped": 0, "gate_calls": 0, "gate_cached": 0, "gate_unparsed": 0}
        self.gate_path = os.path.join(self.out_dir, GATE_CALLS_FILE)
        self.gate_cache = {r["key"]: r for r in read_rows(self.gate_path)}

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
        reports = [self.reports.get(n) for n in chain["path"][1:]]
        app = node_app(env_name=self.env_name, node=self.nodes[child], expansions=self.expansions)
        args = dict(env_description=self.env_description, app_note=pn.app_note(app=app, where="Image 3"),
                    reports=reports, steps=steps, latest=self.reports.get(child),
                    train=self.train_block)
        if self.steps:
            args.update(env_name=self.env_name, windows=self.windows.get(child))
        if chain["task"] is None:
            prompt, allowed = pn.build_prompt(**{"env_name": self.env_name, **args}), (pn.TOO_SHORT, pn.TASK)
        else:
            prompt, allowed = pn.extend_prompt(**args, current=chain["task"]), (pn.CONTINUE, pn.REVISE, pn.NEW)
        images = [self.frame(nid=chain["path"][0]), self.frame(nid=chain["path"][-1]), self.frame(nid=child)]
        out = str(self.vlm.infer(texts=prompt, images=images, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=0.0)["output"])
        verdict, task, parsed = pn.parse_verdict(text=out, allowed=allowed)
        partial = pn.parse_partial(text=out) if verdict == pn.TOO_SHORT else None
        row = {"key": key, "verdict": verdict, "task": task, "partial": partial, "parsed": parsed, "raw": out}
        if self.steps:      # the action after which the named task (or partial) was complete; the edge's end if none
            row["step"] = pn.parse_step(text=out, n_steps=len(steps))
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
        # where the task named on this edge ended: the edge's end node and the step inside it (STEP_ENVS; else the end)
        step = row.get("step", n_steps)
        if verdict == pn.TOO_SHORT and row.get("partial"):
            # the best task the chain has completed so far: kept if the chain never names a full one
            nxt["partial"] = (row["partial"], child, step)
        if verdict == pn.TASK:
            nxt["task"], nxt["names"] = row["task"], [(row["task"], child, step)]
        elif verdict == pn.REVISE and _same(a=row["task"], b=chain["task"]):
            # the same name again: the task was achieved further on; its end moves, no nested copy
            nxt["names"][-1] = (nxt["names"][-1][0], child, step)
        elif verdict == pn.REVISE:
            nxt["task"] = row["task"]
            nxt["names"].append((row["task"], child, step))
        # CONTINUE (no effect on the task): the chain goes on, the task's end stays where it was achieved
        kids = self.kids.get(child) or []
        if not kids:
            return [], [nxt]
        return [(nxt, k) for k in kids], []

    def task_row(self, *, chain: dict, name: str, end: int, step: int, revisions: list, nested: bool,
                 partial: bool) -> dict:
        """A task row; it ends at action `step` of the edge into `end` ("end_step"; "mid_edge" if before the edge's
        last action). "replay": restore "from_node" (the last edge's start) and take "actions" (the edge's first
        `step` actions) to reach the end state; "restorable": from_node is not flagged unrestorable."""
        path = chain["path"][:chain["path"].index(end) + 1]
        edge = self.edge_steps(nid=end)
        flags = self.nodes[path[-2]].get("flags") or {}
        return {"start": chain["path"][0], "task": name, "end": end, "path": path, "n_edges": len(path) - 1,
                "n_steps": sum(len(self.edge_steps(nid=n)) for n in path[1:-1]) + step, "end_step": step,
                "edge_steps": len(edge), "mid_edge": step < len(edge),
                "replay": {"from_node": path[-2], "actions": [s["action"] for s in edge[:step]]},
                "restorable": not (flags.get("unrestorable") or flags.get("replay_diverged")),
                "revisions": revisions, "nested": nested, "partial": partial}

    def tasks_of(self, *, chain: dict) -> list:
        """The finished chain's task rows: its last name (to its end) and every name a REVISE replaced (nested, to
        where that name ended); a chain without one, its best partial task, if any."""
        if chain["task"] is None:
            if not chain.get("partial"):
                return []
            name, end, step = chain["partial"]
            return [self.task_row(chain=chain, name=name, end=end, step=step, revisions=[name], nested=False,
                                  partial=True)]
        names = chain["names"]
        return [self.task_row(chain=chain, name=name, end=end, step=step, revisions=[x[0] for x in names[:i + 1]],
                              nested=i < len(names) - 1, partial=False)
                for i, (name, end, step) in enumerate(names)]

    # ------------------------------------------------------------------ fit gate
    def gate(self, *, row: dict) -> dict:
        """The cached or new fit-gate verdict on a candidate task row: {"verdict", "failed", "reason", ...}."""
        key = ":".join(str(x) for x in row["path"]) + "|" + row["task"]
        if self.steps:
            key += f"|{row['end_step']}"
        with self._lock:
            hit = self.gate_cache.get(key)
        if hit is not None:
            with self._lock:
                self.counts["gate_cached"] += 1
            return hit
        app = node_app(env_name=self.env_name, node=self.nodes[row["end"]], expansions=self.expansions)
        images = [self.frame(nid=row["start"]), self.frame(nid=row["end"])]
        if self.steps:      # the last edge's screens up to the completing action: its start, then each one's result
            edge = self.edge_steps(nid=row["end"])[:row["end_step"]]
            screens = [self.frame(nid=row["path"][-2])] + [
                FrameRef(path=os.path.join(self.directory, st["frame"])).pil() for st in edge]
            images = [images[0]] + screens[-GATE_MAX_EDGE_SCREENS:]
        prompt = pn.gate_prompt(env_name=self.env_name, env_description=self.env_description,
                                app_note=pn.app_note(app=app, where=f"Image {len(images)}"),
                                reports=[self.reports.get(n) for n in row["path"][1:]], task=row["task"],
                                train=self.train_block, n_last=len(images))
        out = str(self.vlm.infer(texts=prompt, images=images, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=0.0)["output"])
        verdict = {"key": key, **pn.parse_gate(text=out), "raw": out}
        with self._lock:
            self.counts["gate_calls"] += 1
            self.counts["gate_unparsed"] += verdict["verdict"] is None
            self.gate_cache[key] = verdict
            with open(self.gate_path, "a") as f:
                f.write(json.dumps(verdict) + "\n")
        return verdict

    def run_gate(self, *, rows: list) -> tuple:
        """(accepted rows, rejected rows with the gate's "failed" and "reason"); an unreadable verdict rejects."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            verdicts = list(pool.map(lambda r: self.gate(row=r), rows))
        accepted, rejected = [], []
        for r, v in zip(rows, verdicts):
            if v["verdict"] == pn.ACCEPT:
                accepted.append(r)
            else:
                rejected.append({**r, "failed": v["failed"], "reason": v["reason"] or "unreadable verdict"})
        return accepted, rejected

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
        log_info(f"annotate: fit gate on {len(rows)} candidates", parameters=self._parameters)
        accepted, rejected = self.run_gate(rows=rows)
        n_before = len(accepted)
        if self.steps:
            accepted = merge_paraphrases(rows=accepted)
        for name, out in ((CANDIDATES_FILE, rows), (TASKS_FILE, accepted), (REJECTED_FILE, rejected)):
            with open(os.path.join(self.out_dir, name), "w") as f:
                for r in out:
                    f.write(json.dumps(r) + "\n")
        failed = {k: sum(k in r["failed"] for r in rejected) for k in pn.GATE_CRITERIA}
        summary = {**self.counts, "candidates": len(rows), "candidate_texts": len({r["task"] for r in rows}),
                   "tasks": len(accepted), "distinct_texts": len({r["task"] for r in accepted}),
                   "merged_paraphrases": n_before - len(accepted),
                   "mid_edge": sum(r.get("mid_edge", False) for r in accepted),
                   "rejected": len(rejected), "rejected_failed": failed,
                   "starts": len(self.starts), "starts_with_task": len({r["start"] for r in accepted}),
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
@click.option("--max_starts", default=None, type=int,
              help="Only this many start nodes (a seeded sample), and stage 1 only on the edges they can reach.")
@click.option("--workers", default=16, show_default=True, help="Calls at a time.")
def main(env_name, run_name, model_name, model_backend, vllm_base_url, max_steps, n_per_flavour, sample_seed,
         max_starts, workers):
    from cusi.agents.vlm import AgentVLM
    if not annotated_run(run_name=run_name):
        raise click.UsageError(f"{run_name}: fills, cuts, smoke and test runs ({NOT_ANNOTATED_SUFFIXES}, "
                               f"{NOT_ANNOTATED_PREFIXES}) are never annotated")
    parameters = load_parameters()
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    vlm = AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                   parameters=parameters)
    only = None
    if max_starts is not None:      # a sample of starts: describe only the edges its chains can reach
        nodes = node_rows(directory=directory)
        kids = children(nodes=nodes)
        starts = start_nodes(nodes=nodes, kids=kids, max_starts=max_starts, seed=sample_seed)
        only = reachable_edges(nodes=nodes, kids=kids, starts=starts, max_steps=max_steps)
    n = describe_edges(directory=directory, env_name=env_name, vlm=vlm, model_name=model_name, workers=workers,
                       parameters=parameters, only=only)
    summary = PathAnnotator(directory=directory, env_name=env_name, vlm=vlm, model_name=model_name,
                            max_steps=max_steps, n_per_flavour=n_per_flavour, sample_seed=sample_seed,
                            workers=workers, max_starts=max_starts, parameters=parameters).run()
    log_info(f"annotate {directory}: {n} transition reports written; {summary}", parameters=parameters)


if __name__ == "__main__":
    main()
