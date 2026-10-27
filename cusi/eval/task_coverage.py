"""
Task coverage of a search run: how close the tasks proposed from an annotated tree (cusi.search.annotate) come to the
benchmark's test tasks. For each test task, its k most similar proposed tasks by embedding cosine (test tasks embedded
as queries, proposed tasks as documents), each scored 1-10 by an LLM judge for being essentially the same task.

Stages, each resumable:
- embed: test set snapshot, deduplicated proposals, embeddings (cached per text), neighbours;
- judge: judge every (test task, neighbour) pair not yet judged, a judge sanity check, then summary;
- summary: per-task metrics and aggregates (overall and per group: Web site, Android app, GameBoy start state).

Output, next to the proposals: <search run>/task_coverage/<embedder>/ (test_tasks.jsonl, proposed.jsonl,
embeddings.npz, neighbours.jsonl) and its judge_<judge>/ (judgments.jsonl, sanity.jsonl, per_task.jsonl,
summary.json).

    python run_eval.py task_coverage --env gameboy --run <run> --embedder_model <id> --judge_model <id> \\
        --stage all --vllm_base_url http://127.0.0.1:<port>/v1/
"""
import hashlib
import json
import os
import random
import re
import click
import numpy as np
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.search.store import node_rows
from cusi.utils.log_handling import log_info
from cusi.utils.paths import search_run

STAGES = ("embed", "judge", "summary", "all")
QUERY_INSTRUCTION = ("Given a task for an agent operating software, retrieve proposed tasks that ask for essentially "
                     "the same thing")
ANDROID_SUITE_SEED = 30       # cusi.eval.android's --task_random_seed: instance 0 is the goal eval runs
N_SANITY = 20

JUDGE_PROMPT = """Two tasks for an agent operating {env_description} are given below: Task A, and Task B, a \
candidate. Rate from 1 to 10 how far Task B is essentially the same task as Task A: an agent that has done B has done \
A, or would only need to change a detail (a value typed, an item or date picked) to do A.

Judge three things in order: (1) the same website, app or place in the game; (2) the same feature or flow of screens;
(3) the same goal and the same specific target.
- 10: the same website, app or place, the same goal and the same target.
- 8-9: the same website, app or place and the same feature, doing every part of what A asks, on another specific \
value (another city, name, item or date).
- 5-7: the same website, app or place, and B goes a good part of the way to A (the same screens), but its goal falls \
short of A's or differs from it.
- 2-4: the same website, app or place but a different feature; or the same kind of action somewhere else.
- 1: unrelated.
A different website, app or place in the game caps the score at 3, however alike the actions are. If A asks for \
several things (find and compare, check and count) and B does only some of them, B gets at most 7. If B is vaguer than \
A and does not name A's target ("whoever is inside", "a Pokémon", "a product"), do not assume it reaches that target: \
B gets at most 7. The same feature means the same screen, menu or tool: a product's page and a comparison of two \
products, or an item list and a character's status page, are different features. Use only what the two texts state: \
do not assume a connection they do not make.

Examples from other environments (not this one):
- A: "On the city library's website, find the central branch's Saturday opening hours." B: "On the city library's \
website, find the east branch's opening hours." -> 9 (same site and feature, another branch)
- A: the same. B: "Find the list of a bookstore chain's shop locations." -> 2 (a different website)
- A: "On the city library's website, find the central branch's Saturday opening hours and whether it has a reading \
room." B: "On the city library's website, find the east branch's opening hours." -> 6 (only one of A's two parts)
- A: "In the weather app, add Lisbon to the saved locations." B: "In the weather app, open the saved locations list." \
-> 6 (the same screens, but B stops before adding a city)
- A: "In the weather app, compare tomorrow's highs in Lisbon and Porto." B: "In the weather app, check tomorrow's \
forecast for Madrid." -> 6 (one forecast, not a comparison of two)
- A: "In the farming game, buy turnip seeds at the general store." B: "Enter the general store." -> 5
- A: "In the farming game, talk to the mayor." B: "Enter the town hall and talk to whoever is inside." -> 6 (B does \
not say it reaches the mayor)
- A: "In the farming game, buy turnip seeds at the general store." B: "Water the crops next to the house." -> 2

Task A: {test}
Task B: {proposed}

Reply in exactly this format:
Reason: <one sentence>
Score: <an integer from 1 to 10>"""

PARAPHRASE_PROMPT = """Rewrite this task for an agent operating {env_description} in different words, keeping exactly \
the same meaning, goal and specific targets. Reply with the rewritten task only.

Task: {task}"""

_SCORE_RE = re.compile(r"score\s*:\s*(\d+)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.*?)(?=\n\s*score\s*:|\Z)", re.IGNORECASE | re.DOTALL)


def text_key(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode()).hexdigest()


def short_name(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- test tasks and proposals

def test_tasks(*, env_name: str, parameters: dict) -> list:
    """The env's test tasks: {"test_id", "task", "group"}."""
    if env_name == "web":
        from cusi.envs.webvoyager import filtered_task_file
        rows = read_jsonl(filtered_task_file(parameters=parameters))
        return [{"test_id": r["id"], "task": r["ques"].strip(), "group": r["web_name"]} for r in rows]
    if env_name == "gameboy":
        from gameboy_worlds.utils import get_benchmark_tasks
        from cusi.agents.specs import GAMEBOY_GAME
        rows = get_benchmark_tasks(game=GAMEBOY_GAME).reset_index(drop=True)
        return [{"test_id": f"{i:03d}", "task": str(r["task"]).strip(), "group": r["init_state"]}
                for i, r in rows.iterrows()]
    if env_name == "android":
        from android_world import registry
        from cusi.envs.android_world import instance_seed
        reg = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
        out = []
        for name in sorted(reg):
            cls = reg[name]
            # android_world's _instantiate_task without the device clock: the seed fixes the goal
            random.seed(instance_seed(suite_seed=ANDROID_SUITE_SEED, task_name=name, instance=0))
            goal = cls(cls.generate_random_params()).goal
            # the template's apps, else its name's leading word (SystemBrightnessMax -> System)
            apps = getattr(cls, "app_names", ()) or (re.match(r"[A-Z][a-z]+", name).group(0).lower(),)
            out.append({"test_id": name, "task": " ".join(str(goal).split()), "group": ",".join(apps)})
        return out
    raise ValueError(f"unknown env {env_name!r}")


def proposed_tasks(*, directory: str) -> list:
    """Every task on the annotated tree's nodes, deduplicated by text: {"proposed_id", "task", "sources"}."""
    nodes = node_rows(directory=directory)
    if not any(n.get("node_notes_raw") is not None for n in nodes.values()):
        raise click.UsageError(f"{directory} has no task annotations: run cusi.search.annotate first")
    by_key = {}
    for nid in sorted(nodes):
        for t in nodes[nid].get("tasks") or []:
            text = " ".join(t["task"].split())
            entry = by_key.setdefault(text_key(text), {"task": text, "sources": []})
            entry["sources"].append({"node_id": nid, "source": t["source"]})
    return [{"proposed_id": f"p{i}", **e} for i, e in enumerate(by_key.values())]


# --------------------------------------------------------------------------- embeddings

class Embedder:
    """A Qwen3-Embedding style model through transformers: last-token pooling, left padding, queries prefixed with
    an instruction, documents as they are; unit vectors."""

    def __init__(self, *, model_name: str, batch_size: int = 16, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.batch_size, self.max_length = torch, batch_size, max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        # bfloat16 where the GPU has it (Ampere on), float16 on older ones (Turing), float32 on the CPU
        dtype = torch.float32 if self.device == "cpu" else \
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device).eval()

    def encode(self, *, texts: list, query: bool) -> np.ndarray:
        from cusi.utils.compute import gpu_work
        torch = self.torch
        texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{t}" if query else t for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad(), gpu_work("embed_tasks"):
                hidden = self.model(**batch).last_hidden_state[:, -1]
            out.append(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def cached_embeddings(*, path: str, texts: list, query: bool, embedder_factory) -> np.ndarray:
    """Vectors for texts (rows in order), reusing path's cache (keyed by role and text) and adding what is new."""
    cache = {}
    if os.path.exists(path):
        with np.load(path) as z:
            cache = {k: z[k] for k in z.files}
    role = "q" if query else "d"
    keys = [f"{role}_{text_key(t)}" for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    if missing:
        vectors = embedder_factory().encode(texts=[t for _, t in missing], query=query)
        cache.update({k: v for (k, _), v in zip(missing, vectors)})
        tmp = f"{path}.tmp.npz"
        np.savez(tmp, **cache)
        os.replace(tmp, path)
    return np.stack([cache[k] for k in keys])


# --------------------------------------------------------------------------- judging

def parse_judgment(text: str) -> tuple:
    """(score 1-10 or None, reason) from a 'Reason: ... Score: N' reply."""
    m = _SCORE_RE.search(text or "")
    score = int(m.group(1)) if m else None
    if score is not None and not 1 <= score <= 10:
        score = None
    r = _REASON_RE.search(text or "")
    return score, " ".join(r.group(1).split()) if r else ""


def judge_pairs(*, vlm, env_description: str, pairs: list, max_new_tokens: int, batch_size: int) -> list:
    """[(test text, proposed text)] -> [{"score", "reason", "raw"}], in batches (one call per batch)."""
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        prompts = [JUDGE_PROMPT.format(env_description=env_description, test=a, proposed=b) for a, b in chunk]
        outputs = vlm.infer(texts=prompts, max_new_tokens=max_new_tokens, temperature=0.0)["output"]
        for raw in outputs:
            raw = str(raw[0] if isinstance(raw, list) else raw)
            score, reason = parse_judgment(raw)
            out.append({"score": score, "reason": reason, "raw": raw})
    return out


# --------------------------------------------------------------------------- stages

def run_embed(*, cov_dir: str, env_name: str, directory: str, embedder_model: str, k: int, parameters: dict) -> None:
    tests = test_tasks(env_name=env_name, parameters=parameters)
    proposals = proposed_tasks(directory=directory)
    write_jsonl(tests, os.path.join(cov_dir, "test_tasks.jsonl"))
    write_jsonl(proposals, os.path.join(cov_dir, "proposed.jsonl"))
    holder = {}

    def factory():
        if "e" not in holder:
            log_info(f"task_coverage: loading {embedder_model}", parameters=parameters)
            holder["e"] = Embedder(model_name=embedder_model)
        return holder["e"]

    path = os.path.join(cov_dir, "embeddings.npz")
    q = cached_embeddings(path=path, texts=[t["task"] for t in tests], query=True, embedder_factory=factory)
    d = cached_embeddings(path=path, texts=[p["task"] for p in proposals], query=False, embedder_factory=factory)
    sims = q @ d.T
    rows = []
    for t, row in zip(tests, sims):
        top = np.argsort(-row)[:k]
        rows.append({"test_id": t["test_id"], "neighbours": [
            {"rank": r + 1, "proposed_id": proposals[j]["proposed_id"], "cos": float(row[j])}
            for r, j in enumerate(top)]})
    write_jsonl(rows, os.path.join(cov_dir, "neighbours.jsonl"))
    with open(os.path.join(cov_dir, "config.json"), "w") as f:
        json.dump({"env": env_name, "run_dir": directory, "embedder_model": embedder_model, "k": k,
                   "query_instruction": QUERY_INSTRUCTION, "n_test": len(tests), "n_proposed": len(proposals),
                   "android_suite_seed": ANDROID_SUITE_SEED if env_name == "android" else None}, f, indent=1)
    log_info(f"task_coverage embed: {len(tests)} test tasks, {len(proposals)} proposed -> {cov_dir}",
             parameters=parameters)


def run_judge(*, cov_dir: str, judge_dir: str, env_name: str, vlm, max_new_tokens: int, batch_size: int,
              parameters: dict) -> None:
    tests = {t["test_id"]: t for t in read_jsonl(os.path.join(cov_dir, "test_tasks.jsonl"))}
    proposals = {p["proposed_id"]: p for p in read_jsonl(os.path.join(cov_dir, "proposed.jsonl"))}
    neighbours = read_jsonl(os.path.join(cov_dir, "neighbours.jsonl"))
    if not neighbours:
        raise click.UsageError(f"no neighbours in {cov_dir}: run --stage embed first")
    env_description = ENV_SPECS[env_name].env_description
    path = os.path.join(judge_dir, "judgments.jsonl")
    done = {r["key"] for r in read_jsonl(path) if r["score"] is not None}
    todo, seen = [], set()
    for n in neighbours:
        for nb in n["neighbours"]:
            a, b = tests[n["test_id"]]["task"], proposals[nb["proposed_id"]]["task"]
            key = text_key(a + "\n" + b)
            if key not in done and key not in seen:
                seen.add(key)
                todo.append((key, a, b))
    log_info(f"task_coverage judge: {len(done)} pairs judged, {len(todo)} to go", parameters=parameters)
    chunk = batch_size * 4      # appended every chunk, so a stopped run keeps what it judged
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        results = judge_pairs(vlm=vlm, env_description=env_description, pairs=[(a, b) for _, a, b in part],
                              max_new_tokens=max_new_tokens, batch_size=batch_size)
        with open(path, "a") as f:
            for (key, a, b), r in zip(part, results):
                f.write(json.dumps({"key": key, "test": a, "proposed": b, **r}) + "\n")
        log_info(f"task_coverage judge: {min(i + chunk, len(todo))}/{len(todo)}", parameters=parameters)
    run_sanity(judge_dir=judge_dir, env_description=env_description, tests=list(tests.values()),
               proposals=list(proposals.values()), vlm=vlm, max_new_tokens=max_new_tokens, batch_size=batch_size,
               parameters=parameters)


def run_sanity(*, judge_dir: str, env_description: str, tests: list, proposals: list, vlm, max_new_tokens: int,
               batch_size: int, parameters: dict) -> None:
    """The judge on N_SANITY test tasks against a paraphrase of themselves (should score high) and a random
    proposed task (should score low)."""
    path = os.path.join(judge_dir, "sanity.jsonl")
    if os.path.exists(path):
        return
    rng = random.Random(0)
    sample = rng.sample(tests, min(N_SANITY, len(tests)))
    prompts = [PARAPHRASE_PROMPT.format(env_description=env_description, task=t["task"]) for t in sample]
    paraphrases = [" ".join(str(o[0] if isinstance(o, list) else o).split())
                   for o in vlm.infer(texts=prompts, max_new_tokens=200, temperature=0.0)["output"]]
    randoms = [rng.choice(proposals)["task"] for _ in sample]
    pairs = [(t["task"], p) for t, p in zip(sample, paraphrases)] + [(t["task"], r) for t, r in zip(sample, randoms)]
    results = judge_pairs(vlm=vlm, env_description=env_description, pairs=pairs, max_new_tokens=max_new_tokens,
                          batch_size=batch_size)
    rows = [{"kind": "paraphrase" if i < len(sample) else "random", "test": a, "proposed": b, **r}
            for i, ((a, b), r) in enumerate(zip(pairs, results))]
    write_jsonl(rows, path)
    log_info(f"task_coverage sanity -> {path}", parameters=parameters)


def run_summary(*, cov_dir: str, judge_dir: str, parameters: dict) -> dict:
    tests = {t["test_id"]: t for t in read_jsonl(os.path.join(cov_dir, "test_tasks.jsonl"))}
    proposals = {p["proposed_id"]: p for p in read_jsonl(os.path.join(cov_dir, "proposed.jsonl"))}
    scores = {r["key"]: r for r in read_jsonl(os.path.join(judge_dir, "judgments.jsonl")) if r["score"] is not None}
    per_task = []
    for n in read_jsonl(os.path.join(cov_dir, "neighbours.jsonl")):
        t = tests[n["test_id"]]
        nbs = []
        for nb in n["neighbours"]:
            b = proposals[nb["proposed_id"]]["task"]
            j = scores.get(text_key(t["task"] + "\n" + b))
            nbs.append({**nb, "task": b, "score": j["score"] if j else None, "reason": j["reason"] if j else None})
        judged = [x["score"] for x in nbs if x["score"] is not None]
        per_task.append({"test_id": t["test_id"], "group": t["group"], "task": t["task"],
                         "top1_cos": nbs[0]["cos"], "mean_cos": float(np.mean([x["cos"] for x in nbs])),
                         "judge_max": max(judged) if judged else None,
                         "judge_mean": float(np.mean(judged)) if judged else None,
                         "judged": len(judged), "neighbours": nbs})
    write_jsonl(per_task, os.path.join(judge_dir, "per_task.jsonl"))

    def aggregate(rows: list) -> dict:
        full = [r for r in rows if r["judged"] == len(r["neighbours"])]
        out = {"n": len(rows), "fully_judged": len(full),
               "top1_cos": float(np.mean([r["top1_cos"] for r in rows])) if rows else None,
               "mean_cos@k": float(np.mean([r["mean_cos"] for r in rows])) if rows else None}
        if full:
            out.update({"judge_max@k": float(np.mean([r["judge_max"] for r in full])),
                        "judge_mean@k": float(np.mean([r["judge_mean"] for r in full])),
                        # >= 8: the same task up to a detail; >= 5: a good part of the way there
                        "coverage@8": float(np.mean([r["judge_max"] >= 8 for r in full])),
                        "coverage@5": float(np.mean([r["judge_max"] >= 5 for r in full]))})
        return out

    groups = sorted({r["group"] for r in per_task})
    sanity = read_jsonl(os.path.join(judge_dir, "sanity.jsonl"))
    summary = {"n_proposed": len(proposals), "overall": aggregate(per_task),
               "groups": {g: aggregate([r for r in per_task if r["group"] == g]) for g in groups},
               "sanity": {kind: {"n": len(s), "mean_score": float(np.mean(s)) if s else None}
                          for kind in ("paraphrase", "random")
                          for s in [[r["score"] for r in sanity if r["kind"] == kind and r["score"] is not None]]}}
    with open(os.path.join(judge_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log_info(f"task_coverage summary ({judge_dir}): {summary['overall']}; sanity {summary['sanity']}",
             parameters=parameters)
    return summary


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "run_name", required=True, help="An annotated search run of this env.")
@click.option("--stage", default="all", show_default=True, type=click.Choice(STAGES))
@click.option("--embedder_model", required=True, help="A Qwen3-Embedding style model (transformers).")
@click.option("--judge_model", default=None, help="The judge (required for the judge and all stages).")
@click.option("--judge_backend", default="vllm", show_default=True)
@click.option("--vllm_base_url", default=None)
@click.option("--k", default=5, show_default=True, help="Proposed tasks per test task.")
@click.option("--batch_size", default=32, show_default=True, help="Judge prompts per call.")
@click.option("--max_new_tokens", default=200, show_default=True)
@click.pass_obj
def command(parameters, env_name, run_name, stage, embedder_model, judge_model, judge_backend, vllm_base_url, k,
            batch_size, max_new_tokens):
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    cov_dir = os.path.join(directory, "task_coverage", short_name(embedder_model))
    os.makedirs(cov_dir, exist_ok=True)
    if stage in ("embed", "all"):
        run_embed(cov_dir=cov_dir, env_name=env_name, directory=directory, embedder_model=embedder_model, k=k,
                  parameters=parameters)
    if stage == "embed":
        return
    if judge_model is None:
        raise click.UsageError("--judge_model is required for the judge, summary and all stages.")
    judge_dir = os.path.join(cov_dir, f"judge_{short_name(judge_model)}")
    os.makedirs(judge_dir, exist_ok=True)
    if stage in ("judge", "all"):
        from cusi.agents.vlm import AgentVLM
        vlm = AgentVLM(model_name=judge_model, model_backend=judge_backend, vllm_base_url=vllm_base_url,
                       parameters=parameters)
        run_judge(cov_dir=cov_dir, judge_dir=judge_dir, env_name=env_name, vlm=vlm, max_new_tokens=max_new_tokens,
                  batch_size=batch_size, parameters=parameters)
    run_summary(cov_dir=cov_dir, judge_dir=judge_dir, parameters=parameters)
