"""GameBoy test-set evaluation through GameBoyPlayEnv(mode="test") (eval_plan.md §3.2).

Reproduces GameBoyRL's `baseline` supervisor arm (run_benchmark.py baseline -> DummySupervisor ->
one executor leg per task, no hint, no self-termination), with any GameBoyRL executor arm:
    tasks      the first --n_tasks rows of get_benchmark_tasks(game) (a prefix, as native)
    env        get_test_environment(row) with headless, max_steps (175, the shell wrapper's
               default) and wait_ticks 20 (benchmark_scripts/common.py forces it)
    executor   any <action>_<history> arm (cusi_practice.executors.gameboy.ARMS), default
               single_visual (CUSI's choice; run_benchmark.py defaults to single_none); max_new_tokens 2000
               (executor_vlm_max_new_tokens), temperature unset (server default), no hint, no
               self-termination check, 4 consecutive invalid decisions end the leg; the
               executor's decision budget is the same max_steps
    success    the leg ended "terminated" (the task's test tracker)
    recorded   n_steps in emulator steps (info["core"]["steps"]), n_invalid, tokens, subgoals,
               plus the action sequence and the LegReport (gzipped pickle) per task

    python run_eval.py gameboy --model_name google/gemma-4-26b-a4b-it --run_name dev --n_tasks 20
"""
import gzip
import os
import pickle
import time
import click
from cusi_utils.log_handling import log_info, log_warn
from cusi_eval.records import EvalRow, EvalRun, call_tokens
from cusi_practice.executors.gameboy import ARMS, DEFAULT_ARM   # imports GameBoyRL: run in its own process

GAMEBOY_WAIT_TICKS = 20          # GameBoyRL benchmark_scripts/common.py run_episode
DEFAULT_MAX_STEPS = 175          # GameBoyRL scripts/core/utils.sh RUN_BENCHMARK_DEFAULTS
EXECUTOR_MAX_NEW_TOKENS = 2000   # GameBoyRL configs/project_vars.yaml executor_vlm_max_new_tokens



def select_tasks(*, game: str, n_tasks) -> list:
    """GameBoyRL common.select_tasks: the first n rows in benchmark order."""
    from gameboy_worlds.utils import get_benchmark_tasks
    tasks = get_benchmark_tasks(game=game).reset_index(drop=True)
    if n_tasks is not None:
        tasks = tasks.head(n_tasks)
    return [dict(row) for _, row in tasks.iterrows()]


def action_sequence(*, report) -> list:
    """The leg's steps in order: each env step's action name (as GameBoyRL's report.action_name gives
    it, e.g. "UP"), "INVALID" for an invalid decision."""
    from cusi_practice.records import InvalidRecord
    return ["INVALID" if isinstance(s, InvalidRecord) else s.action_label() for s in report.steps]


def run_task(*, row: dict, task_id: str, vlm, executor: str, max_steps: int, controller_variant: str,
             save_video: bool, session_name: str, out_dir: str, parameters: dict) -> EvalRow:
    from cusi_envs.gameboy import GameBoyPlayEnv
    from cusi_practice.executors.gameboy import GameBoyExecutor
    from cusi_practice.records import InvalidRecord
    t0 = time.time()
    env = GameBoyPlayEnv(game=row["game"], mode="test", controller_variant=controller_variant, max_steps=max_steps,
                         wait_ticks=GAMEBOY_WAIT_TICKS, save_video=save_video, session_name=session_name,
                         benchmark_row=row, parameters=parameters)
    try:
        obs, info = env.reset()
        agent = GameBoyExecutor(env=env, vlm=vlm, arm=executor, game=row["game"],
                                max_new_tokens=EXECUTOR_MAX_NEW_TOKENS, temperature=vlm_temperature(vlm),
                                parameters=parameters)
        report = agent.run(task=row["task"], hint=None, max_steps=max_steps, obs=obs, info=info,
                           allow_done_check=False, env_name="gameboy")
        last = env._env.get_info()
        n_steps = int(last["core"]["steps"])
        subgoals = last.get("subgoals", {})
    finally:
        env.close()
    with gzip.open(os.path.join(out_dir, f"{task_id}.pkl.gz"), "wb") as f:
        pickle.dump(report, f)
    tin, tout = call_tokens(report)
    return EvalRow(env="gameboy", task_id=task_id, task=row["task"],
                   success=1.0 if report.termination_reason == "terminated" else 0.0,
                   n_steps=n_steps, n_invalid=sum(isinstance(s, InvalidRecord) for s in report.steps),
                   termination_reason=report.termination_reason, input_tokens=tin, output_tokens=tout,
                   seconds=time.time() - t0,
                   extra={"init_state": row["init_state"], "subgoals_reached": list(subgoals.get("completed", [])),
                          "subgoals_all": list(subgoals.get("all", [])), "n_decisions": len(report.steps),
                          "actions": action_sequence(report=report)})


def vlm_temperature(vlm):
    return getattr(vlm, "eval_temperature", None)


@click.command(name="gameboy")
@click.option("--model_name", required=True, help="Model the executor uses (as served, for vllm).")
@click.option("--model_backend", default="vllm", type=click.Choice(["vllm", "openai", "openrouter", "anthropic",
                                                                     "huggingface"]))
@click.option("--vllm_base_url", default=None, help="Default: the CUSI config's vLLM_base_url.")
@click.option("--run_name", required=True)
@click.option("--game", default="pokemon_red")
@click.option("--n_tasks", default=None, type=int, help="First n benchmark tasks (native prefix). Default: all.")
@click.option("--max_steps", default=DEFAULT_MAX_STEPS, show_default=True,
              help="Emulator step limit and executor decision budget (native: both).")
@click.option("--executor", default=DEFAULT_ARM, type=click.Choice(ARMS), show_default=True,
              help="GameBoyRL executor arm <action>_<history>. Native run_benchmark.py defaults to single_none;"
                   " CUSI uses single_visual.")
@click.option("--controller_variant", default="low_level", show_default=True)
@click.option("--temperature", default=None, type=float, help="Default: unset (server default), as native.")
@click.option("--save_video", default=False, type=bool, help="Native default is True; off here to save disk.")
@click.option("--rerun_failed", default=True, type=bool)
@click.pass_obj
def command(parameters, model_name, model_backend, vllm_base_url, run_name, game, n_tasks, max_steps, executor,
            controller_variant, temperature, save_video, rerun_failed):
    """GameBoy benchmark through GameBoyPlayEnv test mode (GameBoyRL baseline arm)."""
    from cusi_practice.vlm import PracticeVLM
    config = dict(env="gameboy", model_name=model_name, model_backend=model_backend, game=game, n_tasks=n_tasks,
                  max_steps=max_steps, executor=executor, controller_variant=controller_variant,
                  temperature=temperature, wait_ticks=GAMEBOY_WAIT_TICKS, max_new_tokens=EXECUTOR_MAX_NEW_TOKENS)
    out_dir = os.path.join(parameters["storage_dir"], "eval", "gameboy", run_name)
    run = EvalRun(directory=out_dir, config=config)
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    vlm = PracticeVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                      parameters=parameters)
    vlm.eval_temperature = temperature
    tasks = select_tasks(game=game, n_tasks=n_tasks)
    log_info(f"GameBoy eval {run_name}: {len(tasks)} tasks, {executor}, max_steps {max_steps} -> {out_dir}",
             parameters=parameters)
    for i, row in enumerate(tasks):
        task_id = f"{i:03d}"
        if run.done(task_id, rerun_failed=rerun_failed):
            continue
        session = f"cusi_eval_{executor}_{controller_variant}_{model_name.split('/')[-1].lower()}_{run_name}/{i:03d}/"
        try:
            result = run_task(row=row, task_id=task_id, vlm=vlm, executor=executor, max_steps=max_steps,
                              controller_variant=controller_variant, save_video=save_video, session_name=session,
                              out_dir=traj_dir, parameters=parameters)
        except Exception as e:   # as native run_episode: one bad task does not end the sweep
            log_warn(f"GameBoy task {task_id} ({row['task']!r}) failed to run: {type(e).__name__}: {e}",
                     parameters=parameters)
            result = EvalRow(env="gameboy", task_id=task_id, task=row["task"], success=None, n_steps=0, n_invalid=0,
                             termination_reason="error", error=f"{type(e).__name__}: {e}"[:500])
        run.write(result)
        log_info(f"[gameboy {task_id}] {row['task']!r}: success {result.success}, {result.termination_reason}, "
                 f"{result.n_steps} emulator steps, {result.n_invalid} invalid", parameters=parameters)
    log_info(f"GameBoy eval {run_name}: {run.summary()}", parameters=parameters)
