"""
Worker pools and checkpoint helpers shared by the practice stages.
"""
import json
import os
import pickle
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from tqdm import tqdm
from cusi.utils.log_handling import log_info, log_warn


TASK_SOURCES = ("zeroshot", "curiosity")


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
    """Run fn(job, worker=<id>) on threads, each with a distinct worker id; returns the failure count."""
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
