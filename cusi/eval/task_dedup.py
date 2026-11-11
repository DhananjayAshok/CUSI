"""
Semantic dedup of a tree's proposed tasks (after cusi.eval.task_coverage.proposed_tasks' exact-text merge): two tasks
are the same task, reworded, when both hold:
- their embeddings (the same document vectors task_coverage / task_diversity compute) have cosine >= cos_threshold;
- their values agree: the same quoted strings and the same numbers, and every capitalised word after the first in
  either task appears (in any case) in the other. This blocks merges that change the key value (Enterprise Server
  3.20 vs 3.22, currency ISK vs GBP, typed "Hello world" vs "Hello there!").

Merges are direct, never through chains: tasks are taken in order (most sources first), each one not yet merged
leads a group, and joins it only the tasks that pass both tests against the leader itself. The leader's text stands
for the group; its sources are the union of the members'; "variants" keeps the members' texts.

Evaluated 2026-11-11 (plans/decisions.md): word overlap (lemma Jaccard or overlap coefficient > 0.7) merged
other-value tasks and, through chains, unrelated ones, and as an extra test it mostly blocked true rewordings and
translations; embeddings alone (cosine >= 0.95) merged ~15% wrong pairs on Web, mostly other-value swaps, which the
values test catches. Runs on CPU over vectors the embed stage already has (no GPU, nothing shared with the judge's
vLLM).
"""
import json
import os
import re
import click
import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
_QUOTED = re.compile(r"[\"“”'‘’«»「」]([^\"“”'‘’«»「」]{1,80})[\"“”'‘’«»「」]")
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")


def _capitalised(*, text: str) -> set:
    words = _WORD.findall(text)
    return {w.lower() for w in words[1:] if w[:1].isupper()}


def values_agree(*, a: str, b: str) -> bool:
    """The same quoted strings and numbers, and each task's capitalised words present in the other."""
    if {q.strip().lower() for q in _QUOTED.findall(a)} != {q.strip().lower() for q in _QUOTED.findall(b)}:
        return False
    if set(_NUMBER.findall(a)) != set(_NUMBER.findall(b)):
        return False
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return _capitalised(text=a) <= wb and _capitalised(text=b) <= wa


def semantic_dedup(*, proposals: list, vectors: np.ndarray, cos_threshold: float, chunk: int = 4096) -> tuple:
    """proposals (proposed_tasks rows, in the order of vectors' unit rows) -> (merged proposals, merge log rows)."""
    n = len(proposals)
    order = sorted(range(n), key=lambda i: (-len(proposals[i]["sources"]), i))
    # candidate pairs above the cosine threshold, found in chunks of rows
    candidates = [[] for _ in range(n)]
    for s in range(0, n, chunk):
        sims = vectors[s:s + chunk] @ vectors.T
        rows, cols = np.nonzero(sims >= cos_threshold)
        for r, c in zip(rows, cols):
            i = s + int(r)
            if int(c) != i:
                candidates[i].append((int(c), float(sims[r, c])))
    leader_of, merged, log = [None] * n, [], []
    for i in order:
        if leader_of[i] is not None:
            continue
        leader_of[i] = i
        group = dict(proposals[i], sources=list(proposals[i]["sources"]), variants=[])
        for j, cos in sorted(candidates[i], key=lambda x: -x[1]):
            if leader_of[j] is not None or not values_agree(a=proposals[i]["task"], b=proposals[j]["task"]):
                continue
            leader_of[j] = i
            group["sources"].extend(proposals[j]["sources"])
            group["variants"].append(proposals[j]["task"])
            log.append({"leader": proposals[i]["proposed_id"], "member": proposals[j]["proposed_id"],
                        "cos": round(cos, 4), "leader_task": proposals[i]["task"], "member_task": proposals[j]["task"]})
        merged.append(group)
    merged.sort(key=lambda p: int(p["proposed_id"][1:]))     # back in proposed_tasks' order
    return merged, log


DEDUP_MODES = ("semantic", "exact")


def dedup_options(fn):
    """--dedup, --dedup_cos for task_coverage and task_diversity."""
    fn = click.option("--dedup_cos", default=0.95, show_default=True,
                      help="semantic dedup: embedding cosine at least this.")(fn)
    return click.option("--dedup", default="semantic", show_default=True, type=click.Choice(DEDUP_MODES),
                        help="exact: identical text only (results in the embedder dir itself, as before); "
                             "semantic: then cusi.eval.task_dedup (results in its own subdir).")(fn)


def stage_dirs(*, stage_dir: str, dedup: str, cos_threshold: float) -> tuple:
    """(embedding cache path, output dir): one embeddings.npz per embedder whatever the dedup, so changing it
    re-embeds nothing; semantic results go to dedup_cos<c>/ so they never mix with exact ones."""
    out = stage_dir if dedup == "exact" else os.path.join(stage_dir, f"dedup_cos{cos_threshold}")
    os.makedirs(out, exist_ok=True)
    return os.path.join(stage_dir, "embeddings.npz"), out


def apply_dedup(*, proposals: list, vectors: np.ndarray, dedup: str, cos_threshold: float, out_dir: str) -> tuple:
    """-> (proposals, their vectors) after the dedup; semantic writes its merges to out_dir/dedup.jsonl."""
    if dedup == "exact":
        return proposals, vectors
    index = {p["proposed_id"]: i for i, p in enumerate(proposals)}
    merged, log = semantic_dedup(proposals=proposals, vectors=vectors, cos_threshold=cos_threshold)
    with open(os.path.join(out_dir, "dedup.jsonl"), "w") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    return merged, vectors[[index[p["proposed_id"]] for p in merged]]
