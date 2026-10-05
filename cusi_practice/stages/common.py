"""Paths, worker pools and checkpoint helpers shared by the practice stages.

Output layout (one directory per environment and model):

    storage_dir/practice/<env>/<model>/[curiosity/]
        proposals.jsonl            propose:  {"scene", "tasks"} per scene
        attempts/                  attempt:  results.csv, legs.pkl, success.json
        guidance.json              guidance: {group: {task, scene, goal_condition, guidance}}
        practice/                  practice: results.csv, <group>_<attempt>.pkl (LegReport), config.json
                                   clean:    paraphrases.json, clean_decisions.csv
        dataset/                   dataset:  train.jsonl, validation.jsonl, images/, stats.json
"""
import json
import os
import pickle
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from tqdm import tqdm
from cusi_utils.log_handling import log_info, log_warn


TASK_SOURCES = ("zeroshot", "curiosity")


class PracticePaths:
    """source "zeroshot": tasks proposed from first frames (propose -> attempt); "curiosity":
    tasks inferred from exploration trajectories (cusi_explore), which write attempts/ under
    <root>/curiosity/ themselves."""

    def __init__(self, *, parameters: dict, env_name: str, model_name: str, source: str = "zeroshot") -> None:
        self.root = os.path.join(parameters["storage_dir"], "practice", env_name, model_name.split("/")[-1])
        if source == "curiosity":
            self.root = os.path.join(self.root, "curiosity")
        os.makedirs(self.root, exist_ok=True)

    @property
    def proposals(self) -> str:
        return os.path.join(self.root, "proposals.jsonl")

    @property
    def attempts_dir(self) -> str:
        return os.path.join(self.root, "attempts")

    @property
    def guidance(self) -> str:
        return os.path.join(self.root, "guidance.json")

    @property
    def practice_dir(self) -> str:
        return os.path.join(self.root, "practice")

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.root, "dataset")


def atomic_json(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def atomic_pickle(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def run_jobs(*, jobs: list, fn: Callable, n_workers: int, desc: str, on_result: Callable,
             parameters: dict, retries: int = 1) -> int:
    """Run fn(job, worker=<id>) over jobs on n_workers threads; each running job holds a
    distinct worker id in [0, n_workers) (one environment instance per id). on_result(job,
    result) runs on the calling thread. A job that raises is retried `retries` times, then
    logged and skipped (it reruns on the next invocation, since it was never checkpointed).
    Returns the number of failures."""
    ids: queue.Queue = queue.Queue()
    for i in range(n_workers):
        ids.put(i)

    def wrapped(job):
        worker = ids.get()
        try:
            for attempt in range(retries + 1):
                try:
                    return fn(job, worker=worker)
                except Exception:
                    if attempt == retries:
                        raise
                    log_warn(f"{desc}: job {job!r} raised; retrying ({attempt + 1}/{retries}):\n"
                             f"{traceback.format_exc()}", parameters=parameters)
        finally:
            ids.put(worker)

    failures = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(wrapped, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            job = futures[future]
            try:
                result = future.result()
            except Exception:
                failures += 1
                log_warn(f"{desc}: job {job!r} failed:\n{traceback.format_exc()}", parameters=parameters)
                continue
            on_result(job, result)
    if failures:
        log_warn(f"{desc}: {failures}/{len(jobs)} jobs failed (rerun to retry them).", parameters=parameters)
    else:
        log_info(f"{desc}: {len(jobs)} jobs done.", parameters=parameters)
    return failures
