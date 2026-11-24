"""
A task-subset dataset (test-set oracle practice): the success tries of chosen benchmark tasks from an attempt_split
run, e.g. the tasks a strong model solves and the base model does not. Rows are made as the dataset stage makes them
(try_rows). labels="judge" counts a try by the practice judge's verdict and cuts it at the judge's safe success point;
labels="gt" counts it by the benchmark's success and keeps it whole (the env ended it at the goal). On Web the
benchmark's success is the try's auto_eval verdict from score_web_tries (web_scores.json in the attempt run), which
must have been run.

No task is held out: val_copy success train rows (seeded) are also written as success/validation, so the fine-tune's
validation loss measures fit. Written in the dataset stage's layout (success/{train,validation}.jsonl, failure/ empty),
so the fine-tune reads it as any practice dataset.
"""
import json
import os
import pickle
from collections import Counter
import numpy as np
from cusi.practice.stages.attempt_test import load_web_scores, try_gt
from cusi.practice.stages.common import atomic_json
from cusi.practice.stages.dataset import try_rows
from cusi.utils.log_handling import log_error, log_info
from cusi.utils.paths import PracticePaths


def _split_id(*, episode_file: str) -> str:
    """The benchmark task of an episode file (<split_id>_r<repeat>.pkl)."""
    return episode_file[:-len(".pkl")].rsplit("_r", 1)[0]


def build_subset_dataset(*, paths: PracticePaths, from_paths: PracticePaths, split_ids: list, labels: str,
                         val_copy: int, safety_margin: int, seed: int, overwrite: bool, parameters: dict,
                         drop_tags: tuple = (), api: bool = True) -> None:
    """split_ids: the benchmark tasks (results.csv split_id, e.g. web.Apple--12); from_paths: the attempt_split run.
    drop_tags: row tags left out (e.g. frame_diff: a reasoning API model's 60-token frame diffs come out cut off).
    api: the run's model is API-served (try_rows: its replies carry reasoning summaries to strip)."""
    out = paths.dataset_dir
    if os.path.exists(os.path.join(out, "stats.json")) and not overwrite:
        log_info(f"subset_dataset: already built ({out}).", parameters=parameters)
        return
    wanted = set(split_ids)
    names = sorted(n for n in os.listdir(from_paths.episodes_dir)
                   if n.endswith(".pkl") and _split_id(episode_file=n) in wanted)
    found = {_split_id(episode_file=n) for n in names}
    if wanted - found:
        log_error(f"subset_dataset: no episodes for {sorted(wanted - found)}", parameters=parameters)
    web_scores = load_web_scores(root=from_paths.root)
    if labels == "gt" and any(n.startswith("web.") for n in names) and not web_scores:
        log_error(f"subset_dataset: Web gt labels need score_web_tries on {from_paths.root} first",
                  parameters=parameters)
    images_dir = os.path.join(out, "images")
    os.makedirs(images_dir, exist_ok=True)
    train, stats, with_rows = [], Counter(), set()
    for name in names:
        with open(os.path.join(from_paths.episodes_dir, name), "rb") as f:
            record = pickle.load(f)
        stats["task_repeats"] += 1
        for entry in record["tries"]:
            stats["tries"] += 1
            if labels == "gt":
                success = try_gt(record=record, entry=entry, web_scores=web_scores)
                entry = {**entry, "verdict": {**entry["verdict"], "safe_success_point": None}}
            else:
                success = entry["verdict"]["success"]
            if not success:
                continue
            stats["success_tries"] += 1
            rows = try_rows(record=record, entry=entry, success=True, safety_margin=safety_margin,
                            images_dir=images_dir, api=api)
            stats["dropped_rows"] += sum(r["tag"] in drop_tags for r in rows)
            rows = [r for r in rows if r["tag"] not in drop_tags]
            if rows:
                with_rows.add(_split_id(episode_file=name))
            train += rows
    if not train:
        log_error(f"subset_dataset: no {labels} success tries in {len(names)} episodes", parameters=parameters)
    order = np.random.default_rng(seed).permutation(len(train))[:val_copy]
    val = [train[i] for i in sorted(order)]
    for kind, split, rows in (("success", "train", train), ("success", "validation", val),
                              ("failure", "train", []), ("failure", "validation", [])):
        os.makedirs(os.path.join(out, kind), exist_ok=True)
        with open(os.path.join(out, kind, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    by_role_tag = Counter(f"{r['role']}/{r['tag']}" for r in train)
    stats = {**stats, "tasks": len(wanted), "tasks_with_rows": len(with_rows), "success_train_rows": len(train),
             "success_validation_rows": len(val), "validation_copied_from_train": len(val),
             "by_role_tag": dict(by_role_tag)}
    atomic_json({**stats, "split_ids": sorted(wanted), "tasks_without_rows": sorted(wanted - with_rows),
                 "labels": labels, "drop_tags": list(drop_tags), "from": from_paths.root, "seed": seed},
                os.path.join(out, "stats.json"))
    log_info(f"subset_dataset: {stats} -> {out}", parameters=parameters)
