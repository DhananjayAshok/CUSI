"""
A focus dataset (GameBoy test-set practice, a diagnostic): the success tries of a few benchmark tasks from test-set
practice (attempt_test), repeated focus_repeat times, mixed with success rows of another practice dataset (mix-in, to
keep the model's other behaviour) and validated on that dataset's held-out success rows.

Can SFT teach the model a handful of tasks it never solves greedily? Every focus success try is a training row (no
held-out focus tries: there are few), so this measures fitting, not generalisation. Mix-in rows are sampled
without replacement (mix_ratio x focus rows, before repeats) from the mix dataset's success train rows, paraphrases
excluded; their images are copied in, so the dataset is self-contained. Rows are written in the dataset stage's
layout (success/{train,validation}.jsonl), so the fine-tune reads it as any practice dataset.
"""
import json
import os
import pickle
import random
import shutil
from collections import Counter
from cusi.practice.stages.common import atomic_json
from cusi.practice.stages.dataset import try_rows
from cusi.utils.log_handling import log_error, log_info
from cusi.utils.paths import PracticePaths


def _read(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _copy_in(*, row: dict, src_dataset: str, out_dataset: str) -> dict:
    """row with its images copied into out_dataset/images/mix/ and its paths pointed there."""
    messages = []
    for msg in row["messages"]:
        content = msg["content"]
        if isinstance(content, list):
            parts = []
            for p in content:
                if p["type"] == "image":
                    rel = os.path.join("images", "mix", os.path.basename(p["image"]))
                    dst = os.path.join(out_dataset, rel)
                    if not os.path.exists(dst):
                        shutil.copyfile(os.path.join(src_dataset, p["image"]), dst)
                    p = {**p, "image": rel}
                parts.append(p)
            msg = {**msg, "content": parts}
        messages.append(msg)
    return {**row, "messages": messages, "mix_from": os.path.basename(os.path.dirname(src_dataset))}


def build_focus_dataset(*, paths: PracticePaths, from_paths: PracticePaths, tasks: list, labels: str,
                        mix_paths: PracticePaths, mix_ratio: float, focus_repeat: int, val_rows: int,
                        safety_margin: int, seed: int, overwrite: bool, parameters: dict) -> None:
    """tasks: benchmark row indices; from_paths: the attempt_test run; mix_paths: a practice run with a dataset."""
    out = paths.dataset_dir
    if os.path.exists(os.path.join(out, "stats.json")) and not overwrite:
        log_info(f"focus_dataset: already built ({out}).", parameters=parameters)
        return
    mix_dataset = mix_paths.dataset_dir
    if not os.path.exists(os.path.join(mix_dataset, "stats.json")):
        log_error(f"focus_dataset: no mix dataset at {mix_dataset}", parameters=parameters)
    images_dir = os.path.join(out, "images")
    os.makedirs(os.path.join(images_dir, "mix"), exist_ok=True)
    wanted = {f"t{i:03d}_" for i in tasks}
    names = sorted(n for n in os.listdir(from_paths.episodes_dir) if n.endswith(".pkl") and n[:5] in wanted)
    focus, stats = [], Counter()
    for name in names:
        with open(os.path.join(from_paths.episodes_dir, name), "rb") as f:
            record = pickle.load(f)
        stats["focus_task_repeats"] += 1
        for entry in record["tries"]:
            success = entry["gt_success"] if labels == "gt" else entry["verdict"]["success"]
            stats["focus_tries"] += 1
            if not success:
                continue
            stats["focus_success_tries"] += 1
            stats["focus_success_tries_real"] += entry["gt_success"]
            stats[f"focus_success_tries_t{record['task']['benchmark_index']:03d}"] += 1
            if labels == "gt":     # the env ended the try at the goal: keep it whole
                entry = {**entry, "verdict": {**entry["verdict"], "safe_success_point": None}}
            focus += try_rows(record=record, entry=entry, success=True, safety_margin=safety_margin,
                              images_dir=images_dir)
    if not focus:
        log_error(f"focus_dataset: no {labels} success tries for tasks {tasks}", parameters=parameters)
    rng = random.Random(seed)
    pool = [r for r in _read(os.path.join(mix_dataset, "success", "train.jsonl")) if "paraphrase_of" not in r]
    mix = rng.sample(pool, min(len(pool), int(round(mix_ratio * len(focus)))))
    val = _read(os.path.join(mix_dataset, "success", "validation.jsonl"))
    val = rng.sample(val, min(len(val), val_rows))
    mix = [_copy_in(row=r, src_dataset=mix_dataset, out_dataset=out) for r in mix]
    val = [_copy_in(row=r, src_dataset=mix_dataset, out_dataset=out) for r in val]
    train = focus * focus_repeat + mix
    rng.shuffle(train)
    for kind, split, rows in (("success", "train", train), ("success", "validation", val),
                              ("failure", "train", []), ("failure", "validation", [])):
        os.makedirs(os.path.join(out, kind), exist_ok=True)
        with open(os.path.join(out, kind, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    stats.update({"focus_rows": len(focus), "focus_repeat": focus_repeat, "mix_rows": len(mix),
                  "success_train_rows": len(train), "success_validation_rows": len(val)})
    atomic_json({**stats, "tasks": tasks, "labels": labels, "from": from_paths.root, "mix": mix_dataset,
                 "mix_ratio": mix_ratio, "seed": seed}, os.path.join(out, "stats.json"))
    log_info(f"focus_dataset: {dict(stats)} -> {out}", parameters=parameters)
