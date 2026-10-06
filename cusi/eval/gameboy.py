"""GameBoy test-set evaluation through GameBoyPlayEnv(mode="test") (plans/eval_plan.md §3.2), under any
GameBoyRL supervisor arm (plans/agents.md).

--supervisor baseline (default) reproduces GameBoyRL's `baseline` arm (run_benchmark.py baseline ->
DummySupervisor -> one executor leg per task, no hint, no self-termination), with any GameBoyRL
executor arm:
    tasks      the first --n_tasks rows of get_benchmark_tasks(game) (a prefix, as native)
    env        get_test_environment(row) with headless, max_steps (175, the shell wrapper's
               default) and wait_ticks 20 (benchmark_scripts/common.py forces it)
    executor   any <action>_<history> arm (cusi.agents.executors.gameboy.ARMS), default
               single_visual (CUSI's choice; run_benchmark.py defaults to single_none); max_new_tokens 2000
               (executor_vlm_max_new_tokens), temperature unset (server default), no hint, no
               self-termination check, 4 consecutive invalid decisions end the leg; the
               executor's decision budget is the same max_steps
    success    the episode ended "terminated" (the task's test tracker)
--supervisor revision / subgoal / info_subgoal_*: GameBoyRL's benchmark_scripts/<arm>.py settings:
    executor max_new_tokens 8000 (its --executor_max_new_tokens), supervisor calls 5000, legs of
    --max_leg_steps 5, the episode budget max_steps, the same env.
Recorded per task: emulator steps (info["core"]["steps"]), invalid decisions, tokens, subgoals,
the action sequence across legs, the supervisor's counts / plan; episodes/<task_id>/ (cusi.eval.episode)
and trajectories/<task_id>.pkl.gz (the SupervisorReport).

    python run_eval.py gameboy --model_name google/gemma-4-26b-a4b-it --run_name dev --n_tasks 20 \
        [--supervisor subgoal] [--workers 4]
"""
import gzip
import os
import pickle
import time
import click
from cusi.utils.log_handling import log_info, log_warn
from cusi.eval.records import EvalRow, EvalRun
from cusi.eval.supervision import (run_episode, run_pool, supervisor_extra, supervisor_options, supervisor_settings,
                                   supervisor_state)
from cusi.agents.executors.gameboy import ARMS, DEFAULT_ARM   # imports GameBoyRL: run in its own process

GAMEBOY_WAIT_TICKS = 20          # GameBoyRL benchmark_scripts/common.py run_episode
DEFAULT_MAX_STEPS = 175          # GameBoyRL scripts/core/utils.sh RUN_BENCHMARK_DEFAULTS
EXECUTOR_MAX_NEW_TOKENS = 2000   # GameBoyRL configs/project_vars.yaml executor_vlm_max_new_tokens
SUPERVISED_EXECUTOR_MAX_NEW_TOKENS = 8000   # benchmark_scripts/{revision,subgoal,info_subgoal}.py


def select_tasks(*, game: str, n_tasks) -> list:
    """GameBoyRL common.select_tasks: the first n rows in benchmark order."""
    from gameboy_worlds.utils import get_benchmark_tasks
    tasks = get_benchmark_tasks(game=game).reset_index(drop=True)
    if n_tasks is not None:
        tasks = tasks.head(n_tasks)
    return [dict(row) for _, row in tasks.iterrows()]


def action_sequence(*, reports: list) -> list:
    """Every leg's steps in order: each env step's action name (as GameBoyRL's report.action_name gives
    it, e.g. "UP"), "INVALID" for an invalid decision."""
    from cusi.agents.records import InvalidRecord
    return ["INVALID" if isinstance(s, InvalidRecord) else s.action_label() for r in reports for s in r.steps]


def run_task(*, row: dict, task_id: str, model_name: str, model_backend: str, vllm_base_url, temperature,
             executor: str, max_steps: int, controller_variant: str, save_video: bool, session_name: str,
             out_dir: str, settings: dict, parameters=None) -> EvalRow:
    """One task (in the parent or a worker process)."""
    from cusi.utils.parameter_handling import load_parameters
    from cusi.envs.gameboy import GameBoyPlayEnv
    from cusi.eval.episode import write_episode
    from cusi.eval.supervision import model
    from cusi.agents.executors.gameboy import GameBoyExecutor
    from cusi.agents.records import InvalidRecord
    parameters = load_parameters(parameters)
    vlm = model(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url)
    max_new_tokens = EXECUTOR_MAX_NEW_TOKENS if settings["supervisor"] == "baseline" else \
        SUPERVISED_EXECUTOR_MAX_NEW_TOKENS
    t0 = time.time()
    env = GameBoyPlayEnv(game=row["game"], mode="test", controller_variant=controller_variant, max_steps=max_steps,
                         wait_ticks=GAMEBOY_WAIT_TICKS, save_video=save_video, session_name=session_name,
                         benchmark_row=row, parameters=parameters)
    try:
        obs, info = env.reset()

        def make_executor(previous_history=None):
            return GameBoyExecutor(env=env, vlm=vlm, arm=executor, game=row["game"], max_new_tokens=max_new_tokens,
                                   temperature=temperature, previous_history=previous_history,
                                   parameters=parameters)

        supervisor, result = run_episode(env_name="gameboy", settings=settings, task=row["task"], env=env, obs=obs,
                                         info=info, make_executor=make_executor, max_steps=max_steps,
                                         run_kwargs={"env_name": "gameboy"}, game=row["game"], parameters=parameters)
        last = env._env.get_info()
        n_steps = int(last["core"]["steps"])
        subgoals = last.get("subgoals", {})
    finally:
        env.close()
    report = result["report"]
    legs = report.executor_reports
    final = report.last_report
    success = 1.0 if final is not None and final.termination_reason == "terminated" else 0.0
    seconds = time.time() - t0
    extra = {"init_state": row["init_state"], "subgoals_reached": list(subgoals.get("completed", [])),
             "subgoals_all": list(subgoals.get("all", [])), "n_decisions": report.n_steps,
             "actions": action_sequence(reports=legs), **supervisor_extra(report=report, result=result)}
    with gzip.open(os.path.join(out_dir, "trajectories", f"{task_id}.pkl.gz"), "wb") as f:
        pickle.dump(report, f)
    write_episode(directory=os.path.join(out_dir, "episodes", task_id), report=report, gameboy=True, meta={
        "task_id": task_id, "task": row["task"], "env": "gameboy", "game": row["game"], "model": model_name,
        "executor": executor, "executor_max_new_tokens": max_new_tokens, "supervisor": settings["supervisor"],
        "supervisor_model": settings["supervisor_model"] if settings["supervisor"] != "baseline" else None,
        "success": success, "termination": final.termination_reason if final else None, "budget": max_steps,
        "emulator_steps": n_steps, "seconds": seconds, "state": supervisor_state(result=result),
        "env_extras": {"subgoals_reached": extra["subgoals_reached"], "subgoals_all": extra["subgoals_all"],
                       "init_state": row["init_state"]}})
    tin = (report.executor_input_tokens or 0) + (report.supervisor_input_tokens or 0)
    tout = (report.executor_output_tokens or 0) + (report.supervisor_output_tokens or 0)
    return EvalRow(env="gameboy", task_id=task_id, task=row["task"], success=success, n_steps=n_steps,
                   n_invalid=sum(isinstance(s, InvalidRecord) for r in legs for s in r.steps),
                   termination_reason=final.termination_reason if final else None, input_tokens=tin,
                   output_tokens=tout, seconds=seconds, extra=extra)


@click.command(name="gameboy")
@click.option("--model_name", required=True, help="Model the executor uses (as served, for vllm).")
@click.option("--model_backend", default="vllm", type=click.Choice(["vllm", "openai", "openrouter", "anthropic",
                                                                     "huggingface"]))
@click.option("--vllm_base_url", default=None, help="Default: the CUSI config's vLLM_base_url.")
@click.option("--run_name", required=True)
@click.option("--game", default="pokemon_red")
@click.option("--n_tasks", default=None, type=int, help="First n benchmark tasks (native prefix). Default: all.")
@click.option("--max_steps", default=DEFAULT_MAX_STEPS, show_default=True,
              help="Emulator step limit and the episode's decision budget (native: both).")
@click.option("--executor", default=DEFAULT_ARM, type=click.Choice(ARMS), show_default=True,
              help="GameBoyRL executor arm <action>_<history>. Native run_benchmark.py defaults to single_none;"
                   " CUSI uses single_visual.")
@click.option("--controller_variant", default="low_level", show_default=True)
@click.option("--temperature", default=None, type=float, help="Default: unset (server default), as native.")
@click.option("--save_video", default=False, type=bool,
              help="GameBoyWorlds' own emulator video (every frame). The step video is always written.")
@click.option("--rerun_failed", default=True, type=bool)
@click.option("--workers", default=1, show_default=True, help="Tasks run in parallel (one emulator each).")
@supervisor_options
@click.pass_obj
def command(parameters, model_name, model_backend, vllm_base_url, run_name, game, n_tasks, max_steps, executor,
            controller_variant, temperature, save_video, rerun_failed, workers, supervisor, supervisor_model,
            supervisor_backend, supervisor_vllm_base_url, max_leg_steps, info_docs):
    """GameBoy benchmark through GameBoyPlayEnv test mode, under a GameBoyRL supervisor arm."""
    settings = supervisor_settings(supervisor=supervisor, supervisor_model=supervisor_model,
                                   supervisor_backend=supervisor_backend,
                                   supervisor_vllm_base_url=supervisor_vllm_base_url, max_leg_steps=max_leg_steps,
                                   info_docs=info_docs, model_name=model_name, model_backend=model_backend,
                                   vllm_base_url=vllm_base_url)
    config = dict(env="gameboy", model_name=model_name, model_backend=model_backend, game=game, n_tasks=n_tasks,
                  max_steps=max_steps, executor=executor, controller_variant=controller_variant,
                  temperature=temperature, wait_ticks=GAMEBOY_WAIT_TICKS, workers=workers,
                  max_new_tokens=EXECUTOR_MAX_NEW_TOKENS if supervisor == "baseline" else
                  SUPERVISED_EXECUTOR_MAX_NEW_TOKENS, **settings)
    out_dir = os.path.join(parameters["storage_dir"], "eval", "gameboy", run_name)
    run = EvalRun(directory=out_dir, config=config)
    os.makedirs(os.path.join(out_dir, "trajectories"), exist_ok=True)
    tasks = select_tasks(game=game, n_tasks=n_tasks)
    log_info(f"GameBoy eval {run_name}: {len(tasks)} tasks, {executor} under {supervisor}, max_steps {max_steps}, "
             f"{workers} worker(s) -> {out_dir}", parameters=parameters)
    jobs = []
    for i, row in enumerate(tasks):
        task_id = f"{i:03d}"
        if run.done(task_id, rerun_failed=rerun_failed):
            continue
        tag = executor if supervisor == "baseline" else f"{supervisor}_{executor}"
        session = f"cusi_eval_{tag}_{controller_variant}_{model_name.split('/')[-1].lower()}_{run_name}/{i:03d}/"
        jobs.append(dict(row=row, task_id=task_id, model_name=model_name, model_backend=model_backend,
                         vllm_base_url=vllm_base_url, temperature=temperature, executor=executor, max_steps=max_steps,
                         controller_variant=controller_variant, save_video=save_video, session_name=session,
                         out_dir=out_dir, settings=settings, parameters=parameters if workers <= 1 else None))

    def on_result(job, result):
        row, task_id = job["row"], job["task_id"]
        if isinstance(result, Exception):   # as native run_episode: one bad task does not end the sweep
            log_warn(f"GameBoy task {task_id} ({row['task']!r}) failed to run: {type(result).__name__}: {result}",
                     parameters=parameters)
            result = EvalRow(env="gameboy", task_id=task_id, task=row["task"], success=None, n_steps=0, n_invalid=0,
                             termination_reason="error", error=f"{type(result).__name__}: {result}"[:500])
        run.write(result)
        log_info(f"[gameboy {task_id}] {row['task']!r}: success {result.success}, {result.termination_reason}, "
                 f"{result.n_steps} emulator steps, {result.n_invalid} invalid, {result.seconds:.0f}s",
                 parameters=parameters)

    t0 = time.time()
    run_pool(jobs=jobs, worker=run_task, workers=workers, on_result=on_result)
    log_info(f"GameBoy eval {run_name}: {run.summary()} ({time.time() - t0:.0f}s for {len(jobs)} tasks)",
             parameters=parameters)
