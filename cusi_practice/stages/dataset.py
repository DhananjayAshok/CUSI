"""Stage 6: build the training dataset (GameBoyRL create_dataset).

One row per kept model call = a chat (role/content messages, images as file paths) + the
target response. GameBoy rows are one user turn (text, then the frame, as the call was sent);
Android rows are M3A's action calls and summary calls (call_type); WebVoyager rows are the
chat up to that call. Hint/guidance blocks are stripped, so inputs match what the evaluation
agent sees. Kept: successful episodes, calls before the judge's safe-success cutoff (+margin),
not REJECTed by clean, and accepted by the environment (stored `valid`, or a finishing action).

Split by episode per task (val_frac, seeded); train rows are augmented with all but the last
paraphrase, validation rows use the reserved last one.

Output in dataset/: train.jsonl, validation.jsonl, images/, stats.json.
"""
import json
import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from cusi_utils.log_handling import log_info, log_warn
from cusi_practice.executors.base import strip_hint_blocks, strip_hint_messages
from cusi_practice.stages.clean import candidate_calls, episode_cutoff, select_successful
from cusi_practice.stages.common import PracticePaths, atomic_json, load_json


def _replace_task(messages: list, task: str, new: str) -> list:
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({**msg, "content": content.replace(task, new)})
        else:
            out.append({**msg, "content": [{**p, "text": p["text"].replace(task, new)} if p["type"] == "text"
                                           else dict(p) for p in content]})
    return out


def call_to_messages(*, call, image_paths: list) -> list:
    """The call's input as a chat with image paths, hint blocks stripped."""
    if call.messages is None:
        return [{"role": "user", "content": [{"type": "text", "text": strip_hint_blocks(call.prompt)}]
                 + [{"type": "image", "image": p} for p in image_paths]}]
    out = []
    for msg in strip_hint_messages(call.messages):
        content = msg["content"]
        if isinstance(content, str):
            out.append(dict(msg))
        else:
            out.append({"role": msg["role"], "content": [
                {"type": "image", "image": image_paths[p["image"]]} if p["type"] == "image" else dict(p)
                for p in content]})
    return out


def split_episodes(*, episodes_by_task: dict, val_frac: float, seed: int) -> set:
    """_split_episodes_by_task."""
    rng = np.random.default_rng(seed)
    val = set()
    for task, episodes in episodes_by_task.items():
        episodes = sorted(episodes)
        n_val = int(len(episodes) * val_frac)
        if n_val < 1:
            continue
        for i in rng.permutation(len(episodes))[:n_val]:
            val.add(episodes[i])
    return val


def build_dataset(*, spec, paths: PracticePaths, safety_margin: int, val_frac: float, seed: int, overwrite: bool,
                  parameters: dict) -> None:
    practice_dir, out_dir = paths.practice_dir, paths.dataset_dir
    train_path, val_path = os.path.join(out_dir, "train.jsonl"), os.path.join(out_dir, "validation.jsonl")
    if os.path.exists(train_path) and not overwrite:
        log_info(f"dataset: already exists ({train_path}).", parameters=parameters)
        return
    results_csv = os.path.join(practice_dir, "results.csv")
    if not os.path.exists(results_csv):
        log_warn("dataset: no practice/results.csv; run practice first.", parameters=parameters)
        return
    try:
        df = pd.read_csv(results_csv)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame([])
    successful = select_successful(df)
    decisions_path = os.path.join(practice_dir, "clean_decisions.csv")
    if len(successful) and not os.path.exists(decisions_path):
        log_warn("dataset: no clean_decisions.csv; run the clean stage first.", parameters=parameters)
        return
    decisions = {}
    if os.path.exists(decisions_path):
        try:
            for _, r in pd.read_csv(decisions_path).iterrows():
                decisions[(str(r["group_idx"]), int(r["attempt"]), int(r["call_idx"]))] = bool(r["accept"])
        except pd.errors.EmptyDataError:
            pass
    paraphrases = load_json(os.path.join(practice_dir, "paraphrases.json"), {})
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows, episodes_by_task = [], defaultdict(set)
    stats = defaultdict(int)
    stats["episodes"], stats["successful_episodes"] = len(df), len(successful)
    for _, row in successful.iterrows():
        group, attempt = str(row["group_idx"]), int(row["attempt"])
        pkl = os.path.join(practice_dir, f"{group}_{attempt}.pkl")
        if not os.path.exists(pkl):
            stats["missing_pkls"] += 1
            continue
        with open(pkl, "rb") as f:
            report = pickle.load(f)
        cutoff = episode_cutoff(safe_success_point=row.get("safe_success_point"), n_calls=len(report.calls),
                                safety_margin=safety_margin)
        stats["calls_before_cutoff"] += sum(c.tag in spec.dataset_tags for c in report.calls[:cutoff])
        for call_idx, call in candidate_calls(report=report, cutoff=cutoff, dataset_tags=spec.dataset_tags):
            if decisions.get((group, attempt, call_idx)) is False:
                stats["rejected"] += 1
                continue
            image_paths = []
            for i, image in enumerate(call.images):
                path = os.path.join(images_dir, f"{group}_{attempt}_{call_idx}_{i}.png")
                with open(path, "wb") as f:
                    f.write(image.png)
                image_paths.append(path)
            episodes_by_task[row["task_string"]].add((group, attempt))
            rows.append({"episode_id": f"{group}_{attempt}", "env": spec.name, "scene": row["scene"],
                         "task_string": row["task_string"], "call_type": call.tag, "call_idx": call_idx,
                         "messages": call_to_messages(call=call, image_paths=image_paths),
                         "response": call.response})
    stats["kept_calls"] = len(rows)
    stats["dropped_illegal_or_aux"] = stats["calls_before_cutoff"] - stats["rejected"] - len(rows)

    val_episodes = split_episodes(episodes_by_task=episodes_by_task, val_frac=val_frac, seed=seed)
    train_rows, val_rows = [], []
    for row in rows:
        task = row["task_string"]
        task_paraphrases = paraphrases.get(task, [])
        group, attempt = row["episode_id"].rsplit("_", 1)
        if (group, int(attempt)) in val_episodes:
            if task_paraphrases:
                new = task_paraphrases[-1]
                val_rows.append({**row, "task_string": new, "messages": _replace_task(row["messages"], task, new)})
            else:
                val_rows.append(row)
        else:
            train_rows.append(row)
            for new in task_paraphrases[:-1]:
                train_rows.append({**row, "task_string": new, "messages": _replace_task(row["messages"], task, new)})
    for path, split in ((train_path, train_rows), (val_path, val_rows)):
        with open(path, "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
    stats["train_rows"], stats["validation_rows"] = len(train_rows), len(val_rows)
    stats["validation_episodes"] = len(val_episodes)
    atomic_json(dict(stats), os.path.join(out_dir, "stats.json"))
    log_info(f"dataset: {dict(stats)} -> {out_dir}", parameters=parameters)
