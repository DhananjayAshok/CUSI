"""
The SFT datasets, built from the attempt stage's pickles:
- success/: each successful task's successful try, cut at the judge's safe success point (+ safety_margin executor
  calls);
- failure/: every failed try, including those of tasks that succeeded later.

A row is one model call the agent made, executor or supervisor, in the order made:
    {episode_id, task_id, task, task_source, scene, node_id, try, role, tag, call_idx, accepted, messages, response}
messages is the call's input as a chat (text, then its images as paths under images/), with practice's guidance cut
(strip_guidance), so the agent learns to act without it. In success/, executor decisions the env rejected are left
out (accepted False; failure/ keeps them). Judge and critique calls are not the agent's and are never rows.

Tasks are split into train and validation (val_frac, seeded) alike for both datasets, so validation tasks are unseen.
Each successful train task is paraphrased by the model (n_paraphrases) and its success rows are repeated with every
paraphrase in place of the task.
"""
import json
import os
import pickle
from collections import defaultdict
import numpy as np
from cusi.agents.executors.base import strip_guidance, strip_guidance_messages
from cusi.agents.supervisors.report import SupervisorCall
from cusi.practice.prompts import AUGMENT_PARAPHRASE_PROMPT, fill
from cusi.practice.stages.common import atomic_json, load_json, run_jobs
from cusi.utils.log_handling import log_info, log_warn
from cusi.utils.parsing import parse_list
from cusi.utils.paths import PracticePaths

SPLITS = ("train", "validation")


def agent_calls(*, report) -> list:
    """(role, tag, call) for every call the agent made, supervisor and executor, in order."""
    out = []
    for event in report.event_log:
        if isinstance(event, SupervisorCall):
            out.append(("supervisor", event.stage, event))
        else:
            out.extend(("executor", c.tag, c) for c in event.report.calls)
    return out


def success_cutoff(*, safe_success_point, n_executor_calls: int, safety_margin: int) -> int:
    """Leading executor calls to keep: safe_success_point indexes the try's executor calls (judge_leg)."""
    if safe_success_point is None:
        return n_executor_calls
    return min(n_executor_calls, int(safe_success_point) + safety_margin)


def call_messages(*, call, image_paths: list) -> list:
    """The call's input as a chat with image paths, guidance cut."""
    if getattr(call, "messages", None) is None:
        return [{"role": "user", "content": [{"type": "text", "text": strip_guidance(call.prompt)}]
                 + [{"type": "image", "image": p} for p in image_paths]}]
    out = []
    for msg in strip_guidance_messages(call.messages):
        content = msg["content"]
        if isinstance(content, str):
            out.append(dict(msg))
        else:
            out.append({"role": msg["role"], "content": [
                {"type": "image", "image": image_paths[p["image"]]} if p["type"] == "image" else dict(p)
                for p in content]})
    return out


def replace_task(*, messages: list, task: str, new: str) -> list:
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({**msg, "content": content.replace(task, new)})
        else:
            out.append({**msg, "content": [{**p, "text": p["text"].replace(task, new)} if p["type"] == "text"
                                           else dict(p) for p in content]})
    return out


def try_rows(*, record: dict, entry: dict, success: bool, safety_margin: int, images_dir: str) -> list:
    task = record["task"]
    episode_id = f"{task['task_id']}_t{entry['try']}"
    calls = agent_calls(report=entry["report"])
    cutoff = success_cutoff(safe_success_point=entry["verdict"]["safe_success_point"],
                            n_executor_calls=sum(r == "executor" for r, _, _ in calls),
                            safety_margin=safety_margin) if success else None
    rows, n_executor = [], 0
    for call_idx, (role, tag, call) in enumerate(calls):
        accepted = None
        if role == "executor":
            if cutoff is not None and n_executor >= cutoff:
                break
            n_executor += 1
            if call.decision is not None:
                accepted = call.accepted_by_env()
                if success and not accepted:
                    continue
        image_paths = []
        for i, image in enumerate(call.images):
            path = os.path.join(images_dir, f"{episode_id}_{call_idx}_{i}.png")
            with open(path, "wb") as f:
                f.write(image.png)
            image_paths.append(path)
        rows.append({"episode_id": episode_id, "task_id": task["task_id"], "task": task["task"],
                     "task_source": task["source"], "scene": task["scene"], "node_id": task["node_id"],
                     "try": entry["try"], "role": role, "tag": tag, "call_idx": call_idx, "accepted": accepted,
                     "messages": call_messages(call=call, image_paths=image_paths), "response": call.response})
    return rows


def split_tasks(*, task_ids: list, val_frac: float, seed: int) -> set:
    """The validation tasks: a seeded val_frac of them."""
    task_ids = sorted(task_ids)
    order = np.random.default_rng(seed).permutation(len(task_ids))
    return {task_ids[i] for i in order[:int(len(task_ids) * val_frac)]}


def paraphrase(*, spec, vlm, tasks: list, n: int, path: str, max_new_tokens: int, parameters: dict) -> dict:
    """task -> its paraphrases, cached in path."""
    out = load_json(path, {})

    def one(task, *, worker: int) -> list:
        output = vlm.infer(texts=fill(AUGMENT_PARAPHRASE_PROMPT, domain=spec.domain, CORE_TASK=task, N=n),
                           max_new_tokens=max_new_tokens)["output"]
        return [p for p in parse_list(output) if p and p != task][:n]

    def save(task, result):
        if len(result) < n:
            log_warn(f"dataset: only {len(result)} paraphrase(s) for {task!r}.", parameters=parameters)
        out[task] = result
        atomic_json(out, path)

    run_jobs(jobs=[t for t in tasks if t not in out], fn=one, n_workers=8, desc="paraphrase", on_result=save,
             parameters=parameters)
    return out


def build_dataset(*, spec, vlm, paths: PracticePaths, n_paraphrases: int, safety_margin: int, val_frac: float,
                  seed: int, max_new_tokens: int, overwrite: bool, parameters: dict) -> None:
    out_dir = paths.dataset_dir
    if os.path.exists(os.path.join(out_dir, "stats.json")) and not overwrite:
        log_info(f"dataset: already built ({out_dir}).", parameters=parameters)
        return
    if not os.path.isdir(paths.episodes_dir):
        log_warn("dataset: no episodes; run the attempt stage first.", parameters=parameters)
        return
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(paths.episodes_dir) if f.endswith(".pkl"))
    task_ids = [n[:-len(".pkl")] for n in names]
    val = split_tasks(task_ids=task_ids, val_frac=val_frac, seed=seed)
    rows = {(kind, split): [] for kind in ("success", "failure") for split in SPLITS}
    stats = defaultdict(int)
    for name in names:
        with open(os.path.join(paths.episodes_dir, name), "rb") as f:
            record = pickle.load(f)
        stats["tasks"] += 1
        stats["unrestorable"] += record["restore_error"] is not None
        split = "validation" if record["task"]["task_id"] in val else "train"
        for entry in record["tries"]:
            success = entry["verdict"]["success"]
            kind = "success" if success else "failure"
            stats[f"{kind}_tries"] += 1
            rows[(kind, split)] += try_rows(record=record, entry=entry, success=success,
                                            safety_margin=safety_margin, images_dir=images_dir)
    train_tasks = sorted({r["task"] for r in rows[("success", "train")]})
    paraphrases = paraphrase(spec=spec, vlm=vlm, tasks=train_tasks, n=n_paraphrases,
                             path=os.path.join(out_dir, "paraphrases.json"), max_new_tokens=max_new_tokens,
                             parameters=parameters) if n_paraphrases > 0 else {}
    augmented = []
    for r in rows[("success", "train")]:
        for new in paraphrases.get(r["task"], []):
            augmented.append({**r, "task": new, "paraphrase_of": r["task"],
                              "messages": replace_task(messages=r["messages"], task=r["task"], new=new)})
    rows[("success", "train")] += augmented
    for (kind, split), split_rows in rows.items():
        os.makedirs(os.path.join(out_dir, kind), exist_ok=True)
        with open(os.path.join(out_dir, kind, f"{split}.jsonl"), "w") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")
        stats[f"{kind}_{split}_rows"] = len(split_rows)
    stats["paraphrase_rows"] = len(augmented)
    stats["validation_tasks"] = len(val)
    atomic_json(dict(stats), os.path.join(out_dir, "stats.json"))
    log_info(f"dataset: {dict(stats)} -> {out_dir}", parameters=parameters)
