"""What the three runners share for --supervisor (plans/agents.md section 3) and --workers.

    @supervisor_options                    --supervisor, --supervisor_model, --supervisor_backend,
                                           --supervisor_vllm_base_url, --max_leg_steps, --info_docs
    run_episode(...)                       builds the arm's supervisor over a fresh-executor factory
                                           and runs it; returns (SupervisorReport, extras)
    supervisor_extra(report, extras)       the results-row fields every arm adds
    run_pool(...)                          tasks in worker processes (each its own env and model
                                           client), results handed back to the parent in order of
                                           completion; workers=1 runs inline

The supervisor model defaults to the executor's (the value given to --model_name, as GameBoyRL's
run_benchmark.py does), on the same server.
"""
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional
import click
from cusi.agents.supervisors import SUPERVISORS, build_supervisor
from cusi.agents.supervisors.prompts import DOMAINS
from cusi.agents.supervisors.report import jsonable

SUPERVISED_ARMS = tuple(SUPERVISORS)


def supervisor_options(command):
    for option in reversed([
        click.option("--supervisor", default="baseline", type=click.Choice(SUPERVISED_ARMS), show_default=True,
                     help="GameBoyRL supervisor arm wrapping the executor (plans/agents.md)."),
        click.option("--supervisor_model", default=None,
                     help="The supervisor's model. Default: the executor's (--model_name), same server."),
        click.option("--supervisor_backend", default=None, help="Default: the executor's backend."),
        click.option("--supervisor_vllm_base_url", default=None, help="Default: the executor's server."),
        click.option("--max_leg_steps", default=10, show_default=True,
                     help="Steps per supervised leg (GameBoyRL: 5)."),
        click.option("--info_docs", default=None, help="info_subgoal_retrieval: comma-separated document paths."),
    ]):
        command = option(command)
    return command


def supervisor_settings(*, supervisor: str, supervisor_model: Optional[str], supervisor_backend: Optional[str],
                        supervisor_vllm_base_url: Optional[str], max_leg_steps: int, info_docs: Optional[str],
                        model_name: str, model_backend: str, vllm_base_url: Optional[str]) -> dict:
    """The resolved supervisor settings (for config.json and the workers)."""
    return {"supervisor": supervisor, "supervisor_model": supervisor_model or model_name,
            "supervisor_backend": supervisor_backend or model_backend,
            "supervisor_vllm_base_url": supervisor_vllm_base_url or vllm_base_url,
            "max_leg_steps": max_leg_steps, "info_docs": info_docs}


@functools.lru_cache(maxsize=None)
def _vlm(model_name: str, model_backend: str, vllm_base_url: Optional[str]):
    from cusi.agents.vlm import AgentVLM
    return AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url)


def model(*, model_name: str, model_backend: str, vllm_base_url: Optional[str]):
    """One model client per (model, backend, server) per process."""
    return _vlm(model_name, model_backend, vllm_base_url)


def run_episode(*, env_name: str, settings: dict, task: str, env, obs: dict, info: dict, make_executor: Callable,
                max_steps: int, run_kwargs: dict, game: str = "", parameters: dict) -> tuple:
    """Build the arm's supervisor and run it. Returns (supervisor, result dict)."""
    name = settings["supervisor"]
    vlm = None
    if name != "baseline":
        vlm = model(model_name=settings["supervisor_model"], model_backend=settings["supervisor_backend"],
                    vllm_base_url=settings["supervisor_vllm_base_url"])
    documents = None
    if name.startswith("info_subgoal"):
        from cusi.agents.supervisors.info_subgoal import load_documents
        documents = load_documents(env=env_name, mode=name.rsplit("_", 1)[1], info_docs=settings["info_docs"],
                                   game=game, knowledge_vlm=vlm, parameters=parameters)
    supervisor = build_supervisor(name=name, task=task, env=env, domain=DOMAINS[env_name], make_executor=make_executor,
                                  obs=obs, info=info, max_steps=max_steps, run_kwargs=run_kwargs, vlm=vlm, game=game,
                                  max_leg_steps=settings["max_leg_steps"], documents=documents,
                                  parameters=parameters)
    return supervisor, supervisor.evaluate()


def supervisor_extra(*, report, result: dict) -> dict:
    """Results-row fields every arm adds: counts, tokens split, timings, and the arm's state."""
    state = {k: v for k, v in result.items() if k != "report"}
    step_log = state.get("step_log") or []
    attempts = [a for r in step_log for a in r.get("attempts", [])]
    out = {**report.counts(), **{k: round(v, 2) for k, v in report.timing().items()},
           "n_attempts": len(attempts) or len(report.legs),
           "n_targets_cleared": sum(1 for r in step_log if r.get("cleared"))}
    for key in ("plan", "original_plan", "planned", "n_replans", "selected_ids"):
        if key in state:
            out[key] = jsonable(state[key])
    return out


def supervisor_state(*, result: dict) -> dict:
    """The arm's full state for meta.json (step_log, plan, ...)."""
    return jsonable({k: v for k, v in result.items() if k != "report"})


def run_pool(*, jobs: list, worker: Callable, workers: int, on_result: Callable) -> None:
    """worker(**job) for every job; on_result(job, result) in the parent. A worker that raises hands
    the exception back as the result."""
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
