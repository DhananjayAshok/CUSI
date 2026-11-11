"""
GameBoy test-set evaluation through GameBoyPlayEnv test mode, with the final GameBoy agent (cusi/agents/agent.py).
Settings mirror GameBoyRL's benchmark scripts.
"""
import gzip
import os
import pickle
import time
import click
from cusi.utils.log_handling import log_info, log_warn
from cusi.eval.records import EvalRow, EvalRun
from cusi.utils.compute import tally
from cusi.utils.run_dir import run_options
from cusi.utils.paths import eval_episode, eval_run, eval_trajectories
from cusi.eval.supervision import DECODING, model, run_pool, supervisor_extra, supervisor_state
from cusi.eval.options import seed_option, task_list
from cusi.agents.agent import AGENT_VERSION, make_agent

GAMEBOY_WAIT_TICKS = 20          # GameBoyRL benchmark_scripts/common.py run_episode
MAX_STEPS = 175                  # GameBoyRL scripts/core/utils.sh RUN_BENCHMARK_DEFAULTS
CONTROLLER_VARIANT = "low_level"


def select_tasks(*, game: str, n_tasks) -> list:
    """GameBoyRL common.select_tasks: the first n rows in benchmark order."""
    from gameboy_worlds.utils import get_benchmark_tasks
    tasks = get_benchmark_tasks(game=game).reset_index(drop=True)
    if n_tasks is not None:
        tasks = tasks.head(n_tasks)
    return [dict(row) for _, row in tasks.iterrows()]


def select_task_ids(*, task_ids: list) -> list:
    """Rows for ids like pokemon_red_003 (game, then the benchmark row index), each with its task_id."""
    from gameboy_worlds.utils import get_benchmark_tasks
    tables, out = {}, []
    for task_id in task_ids:
        game, index = task_id.rsplit("_", 1)
        if game not in tables:
            tables[game] = get_benchmark_tasks(game=game).reset_index(drop=True)
        out.append({**dict(tables[game].iloc[int(index)]), "task_id": task_id})
    return out


def action_sequence(*, reports: list) -> list:
    """Every leg's action names in order, "INVALID" for an invalid decision."""
    from cusi.agents.records import InvalidRecord
    return ["INVALID" if isinstance(s, InvalidRecord) else s.action_label() for r in reports for s in r.steps]


def run_task(*, row: dict, task_id: str, model_name: str, model_backend: str, vllm_base_url, session_name: str,
             out_dir: str, parameters=None, world_action=None, world_action_device: str = "cpu",
             world_action_shadow: bool = True, max_steps: int = MAX_STEPS) -> EvalRow:
    """One task, with its compute (EvalRow.compute)."""
    with tally() as compute:
        result = _run_task(row=row, task_id=task_id, model_name=model_name, model_backend=model_backend,
                           vllm_base_url=vllm_base_url, session_name=session_name, out_dir=out_dir,
                           parameters=parameters, world_action=world_action, world_action_device=world_action_device,
                           world_action_shadow=world_action_shadow, max_steps=max_steps)
    result.compute = compute.snapshot()
    return result


def _run_task(*, row: dict, task_id: str, model_name: str, model_backend: str, vllm_base_url, session_name: str,
              out_dir: str, parameters, world_action, world_action_device: str, world_action_shadow: bool,
              max_steps: int) -> EvalRow:
    from cusi.utils.parameter_handling import load_parameters
    from cusi.envs.gameboy import GameBoyPlayEnv
    from cusi.eval.episode import write_episode
    from cusi.agents.records import InvalidRecord
    parameters = load_parameters(parameters)
    vlm = model(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url)
    wam = None
    if world_action is not None:    # cached per process: one load for every task a worker runs
        from cusi.world_action import load_world_action_model
        wam = load_world_action_model(spec=world_action, device=world_action_device, parameters=parameters)
    t0 = time.time()
    env = GameBoyPlayEnv(game=row["game"], mode="test", controller_variant=CONTROLLER_VARIANT, max_steps=max_steps,
                         wait_ticks=GAMEBOY_WAIT_TICKS, session_name=session_name,
                         benchmark_row=row, parameters=parameters)
    try:
        obs, info = env.reset()
        agent = make_agent(env_name="gameboy", env=env, vlm=vlm, parameters=parameters, world_action=wam,
                           world_action_shadow=world_action_shadow)
        supervisor, result = agent.run(task=row["task"], obs=obs, info=info, max_steps=max_steps)
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
    with gzip.open(os.path.join(eval_trajectories(run_dir=out_dir), f"{task_id}.pkl.gz"), "wb") as f:
        pickle.dump(report, f)
    wam_meta = None
    if wam is not None:
        from cusi.agents.executors.world_action import episode_summary
        wam_meta = {**wam.summary(), "transfer": wam.transfer_kind(game=row["game"]), "device": world_action_device,
                    "shadow_base": world_action_shadow}
        extra.update(episode_summary(legs=legs))
    write_episode(directory=eval_episode(run_dir=out_dir, task_id=task_id), report=report, gameboy=True, meta={
        "task_id": task_id, "task": row["task"], "env": "gameboy", "game": row["game"], "model": model_name,
        "agent": AGENT_VERSION, "executor": agent.executor_class.name, "supervisor": agent.supervisor_class.__name__,
        "success": success, "termination": final.termination_reason if final else None, "budget": max_steps,
        "emulator_steps": n_steps, "seconds": seconds, "state": supervisor_state(result=result),
        "world_action": wam_meta,
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
@click.option("--rerun_failed", default=True, type=bool)
@click.option("--workers", default=1, show_default=True, help="Tasks run in parallel (one emulator each).")
@click.option("--tasks", default=None, help="Task ids <game>_<row>, comma-separated or a file of lines; overrides"
                                            " --game/--n_tasks.")
@click.option("--max_steps", default=MAX_STEPS, show_default=True,
              help="Emulator step budget per task (GameBoyRL's benchmark default is 175).")
@click.option("--world_action", default=None, help="A World Action Model deciding in place of the executor's policy:"
              " its directory, a world model's name under explore/gameboy/world_model, or 'oracle' (the emulator's "
              "own next screens: the upper bound).")
@click.option("--world_action_device", default="cpu", help="Where the world action model runs (vLLM holds the GPUs).")
@click.option("--world_action_shadow", default=True, type=bool, help="Also record (never execute) the base policy's "
              "own decision at every world-action decision: one extra VLM call per decision.")
@seed_option
@run_options
@click.pass_obj
def command(parameters, model_name, model_backend, vllm_base_url, run_name, game, n_tasks, rerun_failed, workers,
            overwrite, ignore_config_violation, tasks, seed, world_action, world_action_device, world_action_shadow,
            max_steps):
    """GameBoy benchmark through GameBoyPlayEnv test mode, with the final GameBoy agent."""
    tasks = task_list(tasks=tasks)
    wam_summary = None
    if world_action is not None:
        from cusi.world_action import load_world_action_model
        wam = load_world_action_model(spec=world_action, device=world_action_device, parameters=parameters)
        wam_summary = {**wam.summary(), "transfer": wam.transfer_kind(game=game)}
        world_action = wam.directory or world_action
    config = dict(env="gameboy", agent=AGENT_VERSION, model_name=model_name, model_backend=model_backend, game=game,
                  n_tasks=n_tasks, tasks=tasks, workers=workers, seed=seed, decoding=DECODING,
                  world_action=world_action, world_action_model=wam_summary,
                  world_action_shadow=world_action_shadow if world_action else None,
                  **({} if max_steps == MAX_STEPS else {"max_steps": max_steps}))
    out_dir = eval_run(parameters=parameters, env="gameboy", run=run_name)
    run = EvalRun(directory=out_dir, config=config, overwrite=overwrite,
                  ignore_config_violation=ignore_config_violation)
    os.makedirs(eval_trajectories(run_dir=out_dir), exist_ok=True)
    rows = select_task_ids(task_ids=tasks.split(",")) if tasks else \
        [{**row, "task_id": f"{i:03d}"} for i, row in enumerate(select_tasks(game=game, n_tasks=n_tasks))]
    log_info(f"GameBoy eval {run_name}: {len(rows)} tasks, {workers} worker(s) -> {out_dir}", parameters=parameters)
    jobs = []
    for row in rows:
        task_id = row.pop("task_id")
        if run.done(task_id, rerun_failed=rerun_failed):
            continue
        session = f"cusi_eval_{AGENT_VERSION}_{model_name.split('/')[-1].lower()}_{run_name}/{task_id}/"
        jobs.append(dict(row=row, task_id=task_id, model_name=model_name, model_backend=model_backend,
                         vllm_base_url=vllm_base_url, session_name=session, out_dir=out_dir,
                         parameters=parameters if workers <= 1 else None,
                         world_action=world_action, world_action_device=world_action_device,
                         world_action_shadow=world_action_shadow, max_steps=max_steps))

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
