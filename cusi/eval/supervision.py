"""
Model clients, worker pools and supervisor record fields shared by the eval runners.
"""
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional
from cusi.agents.supervisors.report import jsonable


@functools.lru_cache(maxsize=None)
def _vlm(model_name: str, model_backend: str, vllm_base_url: Optional[str]):
    from cusi.agents.vlm import AgentVLM
    return AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url)


def model(*, model_name: str, model_backend: str, vllm_base_url: Optional[str]):
    """One model client per (model, backend, server) per process."""
    return _vlm(model_name, model_backend, vllm_base_url)


def supervisor_extra(*, report, result: dict) -> dict:
    """Results-row fields: call/token counts, timing, and the GameBoy plan's numbers (as the E7 runs recorded)."""
    out = {**report.counts(), **{k: round(v, 2) for k, v in report.timing().items()}}
    step_log = result.get("step_log")
    if step_log is not None:
        attempts = [a for r in step_log for a in r.get("attempts", [])]
        out["n_attempts"] = len(attempts) or len(report.legs)
        out["n_targets_cleared"] = sum(1 for r in step_log if r.get("cleared"))
    for key in ("plan", "original_plan", "planned", "n_replans"):
        if key in result:
            out[key] = jsonable(result[key])
    return out


def supervisor_state(*, result: dict) -> dict:
    """The supervisor's state for meta.json (the GameBoy step log and plan)."""
    return jsonable({k: v for k, v in result.items() if k != "report"})


def run_pool(*, jobs: list, worker: Callable, workers: int, on_result: Callable) -> None:
    """worker(**job) per job, on_result(job, result) in the parent; exceptions come back as results."""
    if workers <= 1:
        for job in jobs:
            try:
                result = worker(**job)
            except Exception as e:
                result = e
            on_result(job, result)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(worker, **job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                result = e
            on_result(futures[future], result)
