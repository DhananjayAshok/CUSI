"""
Task diversity of a search run: how many distinct functionalities the tasks proposed from an annotated tree
(cusi.search.annotate) exercise, so that three near-identical tasks count as one and three that test different things
as three. Every proposed task is embedded; each task's k most similar other tasks by embedding cosine are scored 1-10
by an LLM judge (cusi.eval.judge_prompts, internal) for exercising the same functionality, and while all of a task's
judged neighbours score >= SAME_FUNCTION_AT (7), its
next k are judged too (up to k_max): with k = 5 alone, 45% of a GameBoy tree's tasks had all 5 judged the same, which
left the rest of their near-duplicates counted as distinct. Pairs never judged are taken as different (judging all
pairs is ~74M judge calls on a 12k-task Web tree).

Metrics (summary.json), compared within an env only:
- functions@t: the number of distinct functionalities, as the size of a maximal set of tasks no two of which were
  judged >= t (t = 7: fundamentally the same task up to entity names and details; t = 5: similar), greedily chosen;
- vendi: the Vendi score (Friedman & Dieng 2023) of the cosine kernel, the effective number of distinct tasks by
  embedding alone (no judge);
- rarefaction: both at fixed numbers of tasks (random subsets, mean over repeats), since a larger tree finds more
  functionalities just by proposing more tasks.

Stages, each resumable:
- embed: deduplicated proposals, embeddings (cached per text), each task's k_max nearest;
- judge: rounds of neighbouring pairs not yet judged (k per task, then k more for the tasks still all >= 7), a judge
  sanity check, then summary;
- summary: the metrics above.

Proposals are deduplicated as in task_coverage (--dedup semantic, the default, or exact).

Output, next to the proposals: <search run>/task_diversity/<embedder>/ (embeddings.npz, every dedup's cache) and the
dedup's output dir (the embedder dir itself for exact, dedup_cos<c>/ in it for semantic): proposed.jsonl,
dedup.jsonl, neighbours.jsonl, and its judge_<judge>/ (judgments.jsonl, sanity.jsonl, groups.jsonl, summary.json).

    python run_eval.py task_diversity --env gameboy --run <run> --embedder_model <id> --judge_model <id> \\
        --stage all --vllm_base_url http://127.0.0.1:<port>/v1/
"""
import json
import os
import random
import click
import numpy as np
from cusi.agents.specs import ENV_NAMES
from cusi.eval.task_coverage import (PARAPHRASE_PROMPT, Embedder, cached_embeddings, judge_prompts, proposed_tasks,
                                     read_jsonl, short_name, text_key, write_jsonl)
from cusi.eval.judge_prompts import ENV_DESCRIPTIONS, INTERNAL_PROMPTS, SAME_FUNCTION_AT
from cusi.eval.task_dedup import apply_dedup, dedup_options, stage_dirs
from cusi.utils.log_handling import log_info
from cusi.utils.paths import search_run

STAGES = ("embed", "judge", "summary", "all")
THRESHOLDS = (SAME_FUNCTION_AT, 5)
N_SANITY = 20
RAREFACTION_SIZES = (25, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
RAREFACTION_REPEATS = 20

def pair_key(a: str, b: str) -> str:
    """Order-free key of a pair of task texts."""
    return text_key("\n".join(sorted((a, b))))


def distinct_set(*, n: int, edges: list) -> tuple:
    """A maximal set of the n items no two of which are joined by an edge (greedy, fewest edges first) -> (the
    chosen items, each item's representative: itself if chosen, else the first chosen item it is joined to).
    Connected groups were tried first: chains of similar pairs (walk north ~ walk to the building ~ walk to the edge)
    put 657 of 823 GameBoy tasks in one group, and the count fell as tasks were added."""
    adjacent = [[] for _ in range(n)]
    for i, j in edges:
        adjacent[i].append(j)
        adjacent[j].append(i)
    rep = [-1] * n
    chosen = []
    for i in sorted(range(n), key=lambda x: (len(adjacent[x]), x)):
        if rep[i] != -1:
            continue
        rep[i] = i
        chosen.append(i)
        for j in adjacent[i]:
            if rep[j] == -1:
                rep[j] = i
    return chosen, rep


def vendi(*, vectors: np.ndarray) -> float:
    """exp(entropy of the eigenvalues of K / n), K the cosine kernel of unit vectors; through the smaller of the n x n
    and d x d Gram matrices (the same nonzero eigenvalues)."""
    n = len(vectors)
    if n == 0:
        return 0.0
    x = vectors.astype(np.float64)
    gram = x @ x.T if n <= x.shape[1] else x.T @ x
    eig = np.clip(np.linalg.eigvalsh(gram / n), 0.0, None)
    eig = eig[eig > 1e-12]
    eig = eig / eig.sum()
    return float(np.exp(-(eig * np.log(eig)).sum()))


# --------------------------------------------------------------------------- stages

def run_embed(*, div_dir: str, emb_path: str, directory: str, embedder_model: str, k_max: int, dedup: str,
              dedup_cos: float, parameters: dict) -> None:
    proposals = proposed_tasks(directory=directory)
    n_exact = len(proposals)
    holder = {}

    def factory():
        if "e" not in holder:
            log_info(f"task_diversity: loading {embedder_model}", parameters=parameters)
            holder["e"] = Embedder(model_name=embedder_model)
        return holder["e"]

    # Documents (no query instruction) on both sides: a symmetric similarity
    vectors = cached_embeddings(path=emb_path, texts=[p["task"] for p in proposals], query=False,
                                embedder_factory=factory)
    proposals, vectors = apply_dedup(proposals=proposals, vectors=vectors, dedup=dedup, cos_threshold=dedup_cos, out_dir=div_dir)
    write_jsonl(proposals, os.path.join(div_dir, "proposed.jsonl"))
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    top = min(k_max, len(proposals) - 1)
    rows = []
    for p, row in zip(proposals, sims):
        nearest = np.argpartition(-row, top - 1)[:top] if top > 0 else np.array([], dtype=int)
        nearest = nearest[np.argsort(-row[nearest])]
        rows.append({"proposed_id": p["proposed_id"],
                     "neighbours": [[proposals[j]["proposed_id"], float(row[j])] for j in nearest]})
    write_jsonl(rows, os.path.join(div_dir, "neighbours.jsonl"))
    with open(os.path.join(div_dir, "config.json"), "w") as f:
        json.dump({"run_dir": directory, "embedder_model": embedder_model, "k_max": k_max,
                   "n_proposed": len(proposals), "dedup": dedup,
                   "dedup_cos": dedup_cos if dedup == "semantic" else None, "n_exact_text": n_exact}, f,
                  indent=1)
    log_info(f"task_diversity embed: {len(proposals)} proposed tasks, {top} nearest each -> {div_dir}",
             parameters=parameters)


def run_judge(*, div_dir: str, judge_dir: str, env_name: str, vlm, k: int, max_new_tokens: int, batch_size: int,
              parameters: dict) -> None:
    proposals = {p["proposed_id"]: p["task"] for p in read_jsonl(os.path.join(div_dir, "proposed.jsonl"))}
    neighbours = read_jsonl(os.path.join(div_dir, "neighbours.jsonl"))
    if not neighbours:
        raise click.UsageError(f"no neighbours in {div_dir}: run --stage embed first")
    path = os.path.join(judge_dir, "judgments.jsonl")
    scores = {r["key"]: r["score"] for r in read_jsonl(path) if r["score"] is not None}
    depth, frontier = k, neighbours           # each frontier task's first `depth` neighbours get judged
    while frontier:
        todo, seen = [], set()
        for n in frontier:
            a = proposals[n["proposed_id"]]
            for b_id, _ in n["neighbours"][:depth]:
                b = proposals[b_id]
                key = pair_key(a, b)
                if key not in scores and key not in seen:
                    seen.add(key)
                    todo.append((key, a, b))
        log_info(f"task_diversity judge: depth {depth}, {len(frontier)} tasks, {len(scores)} pairs judged, "
                 f"{len(todo)} to go", parameters=parameters)
        chunk = batch_size * 4      # appended every chunk, so a stopped run keeps what it judged
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            prompts = [INTERNAL_PROMPTS[env_name].format(a=a, b=b) for _, a, b in part]
            results = judge_prompts(vlm=vlm, prompts=prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            with open(path, "a") as f:
                for (key, a, b), r in zip(part, results):
                    f.write(json.dumps({"key": key, "a": a, "b": b, **r}) + "\n")
                    if r["score"] is not None:
                        scores[key] = r["score"]
            log_info(f"task_diversity judge: {min(i + chunk, len(todo))}/{len(todo)}", parameters=parameters)
        # Tasks whose judged neighbours so far are all the same functionality: look k further down their list
        frontier = [n for n in frontier
                    if len(n["neighbours"]) > depth and all(
                        scores.get(pair_key(proposals[n["proposed_id"]], proposals[b_id]), 0) >= THRESHOLDS[0]
                        for b_id, _ in n["neighbours"][:depth])]
        depth += k
    run_sanity(judge_dir=judge_dir, env_name=env_name,
               proposals=[{"task": t} for t in proposals.values()], vlm=vlm, max_new_tokens=max_new_tokens,
               batch_size=batch_size, parameters=parameters)


def run_sanity(*, judge_dir: str, env_name: str, proposals: list, vlm, max_new_tokens: int, batch_size: int,
               parameters: dict) -> None:
    """The judge on N_SANITY proposed tasks against a paraphrase of themselves (should score 9+) and another random
    proposed task (should mostly score low)."""
    path = os.path.join(judge_dir, "sanity.jsonl")
    if os.path.exists(path):
        return
    rng = random.Random(0)
    sample = rng.sample(proposals, min(N_SANITY, len(proposals)))
    prompts = [PARAPHRASE_PROMPT.format(env_description=ENV_DESCRIPTIONS[env_name], task=p["task"]) for p in sample]
    paraphrases = [" ".join(str(o[0] if isinstance(o, list) else o).split())
                   for o in vlm.infer(texts=prompts, max_new_tokens=200, temperature=0.0)["output"]]
    randoms = [rng.choice(proposals)["task"] for _ in sample]
    pairs = [(p["task"], q) for p, q in zip(sample, paraphrases)] + [(p["task"], r) for p, r in zip(sample, randoms)]
    results = judge_prompts(vlm=vlm, prompts=[INTERNAL_PROMPTS[env_name].format(a=a, b=b) for a, b in pairs],
                            max_new_tokens=max_new_tokens, batch_size=batch_size)
    rows = [{"kind": "paraphrase" if i < len(sample) else "random", "a": a, "b": b, **r}
            for i, ((a, b), r) in enumerate(zip(pairs, results))]
    write_jsonl(rows, path)
    log_info(f"task_diversity sanity -> {path}", parameters=parameters)


def run_summary(*, div_dir: str, emb_path: str, judge_dir: str, seed: int, parameters: dict) -> dict:
    proposals = read_jsonl(os.path.join(div_dir, "proposed.jsonl"))
    index = {p["proposed_id"]: i for i, p in enumerate(proposals)}
    with np.load(emb_path) as z:
        vectors = np.stack([z[f"d_{text_key(p['task'])}"] for p in proposals])
    scores = {r["key"]: r["score"] for r in read_jsonl(os.path.join(judge_dir, "judgments.jsonl"))
              if r["score"] is not None}
    judged = {}         # (i, j) -> score over the neighbouring pairs judged
    for n in read_jsonl(os.path.join(div_dir, "neighbours.jsonl")):
        i = index[n["proposed_id"]]
        for b_id, _ in n["neighbours"]:
            j = index[b_id]
            s = scores.get(pair_key(proposals[i]["task"], proposals[j]["task"]))
            if s is not None:
                judged[(min(i, j), max(i, j))] = s
    judged = [(i, j, s) for (i, j), s in judged.items()]
    n = len(proposals)

    def functions(subset: np.ndarray, t: int) -> int:
        local = {int(g): k for k, g in enumerate(subset)}
        edges = [(local[i], local[j]) for i, j, s in judged if s >= t and i in local and j in local]
        return len(distinct_set(n=len(subset), edges=edges)[0])

    full = np.arange(n)
    overall = {"n_tasks": n, "n_judged": len(judged),
               **{f"functions@{t}": functions(full, t) for t in THRESHOLDS}, "vendi": vendi(vectors=vectors)}
    rng = np.random.default_rng(seed)
    rarefaction = []
    for m in [s for s in RAREFACTION_SIZES if s < n] + [n]:
        reps = 1 if m == n else RAREFACTION_REPEATS
        subsets = [full if m == n else rng.choice(n, size=m, replace=False) for _ in range(reps)]
        row = {"n_tasks": m}
        for t in THRESHOLDS:
            row[f"functions@{t}"] = float(np.mean([functions(s, t) for s in subsets]))
        row["vendi"] = float(np.mean([vendi(vectors=vectors[s]) for s in subsets]))
        rarefaction.append(row)

    # At the strict threshold: each distinct task with the tasks it stands for, largest first
    _, rep = distinct_set(n=n, edges=[(i, j) for i, j, s in judged if s >= THRESHOLDS[0]])
    groups = {}
    for i, r in enumerate(rep):
        groups.setdefault(r, []).append(proposals[i]["task"])
    write_jsonl([{"size": len(v), "representative": proposals[r]["task"], "tasks": v}
                 for r, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))],
                os.path.join(judge_dir, "groups.jsonl"))
    sanity = read_jsonl(os.path.join(judge_dir, "sanity.jsonl"))
    hist = np.bincount([s for _, _, s in judged], minlength=11)[1:].tolist()
    summary = {"overall": overall, "rarefaction": rarefaction, "score_histogram_1_10": hist,
               "sanity": {kind: {"n": len(s), "mean_score": float(np.mean(s)) if s else None,
                                 f"share_ge_{SAME_FUNCTION_AT}": float(np.mean([x >= SAME_FUNCTION_AT for x in s]))
                                 if s else None}
                          for kind in ("paraphrase", "random")
                          for s in [[r["score"] for r in sanity if r["kind"] == kind and r["score"] is not None]]}}
    with open(os.path.join(judge_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log_info(f"task_diversity summary ({judge_dir}): {overall}; sanity {summary['sanity']}", parameters=parameters)
    return summary


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "run_name", required=True, help="An annotated search run of this env.")
@click.option("--stage", default="all", show_default=True, type=click.Choice(STAGES))
@click.option("--embedder_model", required=True, help="A Qwen3-Embedding style model (transformers).")
@click.option("--judge_model", default=None, help="The judge (required for the judge, summary and all stages).")
@click.option("--judge_backend", default="vllm", show_default=True)
@click.option("--vllm_base_url", default=None)
@click.option("--k", default=5, show_default=True, help="Nearest other tasks judged per task (and per round).")
@click.option("--k_max", default=40, show_default=True,
              help="Most neighbours judged for a task whose judged neighbours are all the same functionality.")
@click.option("--batch_size", default=32, show_default=True, help="Judge prompts per call.")
@click.option("--max_new_tokens", default=200, show_default=True)
@click.option("--seed", default=0, show_default=True, help="Rarefaction subsets.")
@dedup_options
@click.pass_obj
def command(parameters, env_name, run_name, stage, embedder_model, judge_model, judge_backend, vllm_base_url, k,
            k_max, batch_size, max_new_tokens, seed, dedup, dedup_cos):
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    emb_path, div_dir = stage_dirs(stage_dir=os.path.join(directory, "task_diversity", short_name(embedder_model)),
                                   dedup=dedup, cos_threshold=dedup_cos)
    if stage in ("embed", "all"):
        run_embed(div_dir=div_dir, emb_path=emb_path, directory=directory, embedder_model=embedder_model,
                  k_max=k_max, dedup=dedup, dedup_cos=dedup_cos, parameters=parameters)
    if stage == "embed":
        return
    if judge_model is None:
        raise click.UsageError("--judge_model is required for the judge, summary and all stages.")
    judge_dir = os.path.join(div_dir, f"judge_{short_name(judge_model)}")
    os.makedirs(judge_dir, exist_ok=True)
    if stage in ("judge", "all"):
        from cusi.agents.vlm import AgentVLM
        vlm = AgentVLM(model_name=judge_model, model_backend=judge_backend, vllm_base_url=vllm_base_url,
                       parameters=parameters)
        run_judge(div_dir=div_dir, judge_dir=judge_dir, env_name=env_name, vlm=vlm, k=k,
                  max_new_tokens=max_new_tokens, batch_size=batch_size, parameters=parameters)
    run_summary(div_dir=div_dir, emb_path=emb_path, judge_dir=judge_dir, seed=seed, parameters=parameters)
