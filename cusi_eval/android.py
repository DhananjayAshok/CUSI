"""AndroidWorld evaluation through AndroidPlayEnv in test mode (eval_plan.md §3.3), equivalent to
android_world/run.py with --agent_name m3a_cusi:

    python run_eval.py android --model_name <served name> --model_backend vllm --run_name dev \
        [--tasks ContactsAddContact,...] [--n_task_combinations 1] [--task_random_seed 30]

Needs the emulator (scripts/android_emulator.sh, inside scripts/container.sh) on --console_port.

Same as the native harness:
- the suite: android_world family tasks (or --tasks), sorted by name, n_task_combinations
  instances each, create_suite's per-instance seeds; one AndroidWorld connection for the suite;
- per task: _instantiate_task, initialize_task, Home if the task starts there, hide the
  automation UI; M3A with M3A's own prompts, temperature 0.0, max_new_tokens 1000; budget
  int(10 * complexity) agent steps, malformed replies included; M3A's observation timing
  (stabilized at each step start, unstabilized right after the 2 s pause for the summary);
- score: task.is_successful if the agent sent `status`, else 0; then task.tear_down;
- an exception (or an emulator hang the env recovered from) marks the task failed-to-run
  (success None), excluded from the success rate, and skips tear_down, as _run_task does;
- metrics: android_world's suite_utils.process_episodes on the same episode fields.

Outputs: storage_dir/eval/android/<run_name>/ results.jsonl (cusi_eval.records),
summary.json, process_episodes.md, trajectories/<task>_<i>.json (every call's prompt tag,
reply and decision).
"""
import json
import os
import time
import traceback
import click
from cusi_utils.log_handling import log_info, log_warn
from cusi_eval.records import EvalRow, EvalRun, call_tokens

ENV_NAME = "android"
MAX_NEW_TOKENS = 1000      # benchmark_adapters/android_world_adapter.py InferenceModelWrapper
TEMPERATURE = 0.0


def suite_tasks(*, tasks: str) -> list:
    """android_world family task names (filtered by --tasks), sorted as create_suite does."""
    from android_world import registry
    names = sorted(registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY))
    if tasks:
        wanted = [t.strip() for t in tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in names]
        if unknown:
            raise click.BadParameter(f"not android_world tasks: {unknown}")
        names = sorted(wanted)
    return names


def trajectory(*, report) -> list:
    return [{"tag": c.tag, "decision": getattr(c, "decision", None), "prompt": c.prompt, "response": c.response,
             "steps": [{"kind": type(s).__name__, "action": getattr(s, "action_text", None),
                        "valid": getattr(s, "valid", None), "error": getattr(s, "error", None),
                        "reason": getattr(s, "reason", None), "reward": getattr(s, "reward", None)}
                       for s in c.steps]} for c in report.calls]


def run_one(*, task_name: str, instance: int, suite_seed: int, connection, emulator, vlm, console_port: int,
            grpc_port: int, parameters: dict) -> tuple:
    """Run one task instance. Returns (row fields, episode dict for process_episodes,
    trajectory, connection to reuse)."""
    from cusi_envs.android_world import AndroidPlayEnv
    from cusi_practice.executors.m3a import M3AExecutor
    start = time.time()
    env, report, goal, complexity_steps = None, None, None, None
    fields = {"success": None, "n_steps": 0, "n_invalid": 0, "termination_reason": None, "error": None,
              "answer": None}
    try:
        env = AndroidPlayEnv(task=task_name, mode="test", suite_seed=suite_seed, instance=instance,
                             console_port=console_port, grpc_port=grpc_port, emulator=emulator,
                             reset_mode="reinit", connection=connection, stabilize_after_action=False,
                             parameters=parameters)
        goal, complexity_steps = env._task.goal, env.max_steps
        recoveries = env.recoveries
        obs, info = env.reset()
        executor = M3AExecutor(env=env, vlm=vlm, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                               refresh_before_decide=True, parameters=parameters)
        report = executor.run(task=goal, hint=None, max_steps=env.max_steps, obs=obs, info=info,
                              max_consecutive_invalid=None, finish_through_env=True, env_name=ENV_NAME)
        fields.update(n_steps=len(report.steps), termination_reason=report.termination_reason,
                      n_invalid=sum(1 for s in report.steps if not getattr(s, "valid", False)),
                      answer=report.answer)
        if env.recoveries > recoveries:
            fields["error"] = "the emulator hung and was restarted mid-task"
        else:
            done = report.termination_reason == "terminated"
            fields["success"] = float(report.final_reward or 0.0) if done else 0.0
            env._guarded(fn=lambda: env._task.tear_down(env._env))
    except Exception:
        fields["error"] = traceback.format_exc()[-2000:]
        log_warn(f"{task_name}_{instance} failed to run:\n{fields['error']}", parameters=parameters)
    finally:
        if env is not None:
            connection = env._env
            env.close(close_connection=False)
    seconds = time.time() - start
    episode = {"goal": goal, "task_template": task_name, "instance_id": instance,
               "is_successful": fields["success"] if fields["success"] is not None else float("nan"),
               "episode_length": fields["n_steps"], "run_time": seconds,
               "exception_info": fields["error"] if fields["success"] is None else None}
    fields["seconds"] = seconds
    fields["goal"] = goal
    fields["budget"] = complexity_steps
    return fields, episode, trajectory(report=report) if report is not None else [], connection, report


@click.command(name="android")
@click.option("--model_name", required=True, help="Model id (as served).")
@click.option("--model_backend", required=True, type=click.Choice(["vllm", "openai", "anthropic", "openrouter",
                                                                   "huggingface"]))
@click.option("--vllm_base_url", default=None)
@click.option("--run_name", required=True)
@click.option("--tasks", default=None, help="Comma-separated task names (default: the whole android_world family).")
@click.option("--n_task_combinations", default=1)
@click.option("--task_random_seed", default=30)
@click.option("--console_port", default=5554)
@click.option("--grpc_port", default=8554)
@click.option("--rerun_failed/--no_rerun_failed", default=True)
@click.pass_obj
def command(parameters, model_name, model_backend, vllm_base_url, run_name, tasks, n_task_combinations,
            task_random_seed, console_port, grpc_port, rerun_failed):
    """AndroidWorld (M3A) on the test suite through AndroidPlayEnv."""
    from cusi_envs.android_world import AndroidEmulator
    from cusi_practice.vlm import PracticeVLM
    from android_world.env import env_launcher
    out_dir = os.path.join(parameters["storage_dir"], "eval", ENV_NAME, run_name)
    run = EvalRun(directory=out_dir, config={
        "env": ENV_NAME, "agent": "m3a", "model_name": model_name, "model_backend": model_backend,
        "tasks": tasks, "n_task_combinations": n_task_combinations, "task_random_seed": task_random_seed,
        "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS})
    os.makedirs(os.path.join(out_dir, "trajectories"), exist_ok=True)
    names = suite_tasks(tasks=tasks)
    vlm = PracticeVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                      parameters=parameters)
    emulator = AndroidEmulator(port=console_port, grpc_port=grpc_port, snapshots=False, parameters=parameters)
    import shutil
    connection = env_launcher.load_and_setup_env(console_port=console_port, grpc_port=grpc_port,
                                                 emulator_setup=False, adb_path=shutil.which("adb"))
    episodes_path = os.path.join(out_dir, "episodes.jsonl")
    for name in names:
        for i in range(n_task_combinations):
            task_id = f"{name}_{i}"
            if run.done(task_id, rerun_failed=rerun_failed):
                continue
            log_info(f"[android eval] {task_id}", parameters=parameters)
            fields, episode, traj, connection, report = run_one(
                task_name=name, instance=i, suite_seed=task_random_seed, connection=connection,
                emulator=emulator, vlm=vlm, console_port=console_port, grpc_port=grpc_port,
                parameters=parameters)
            tokens = call_tokens(report) if report is not None else (0, 0)
            run.write(EvalRow(env=ENV_NAME, task_id=task_id, task=fields["goal"] or name,
                              success=fields["success"], n_steps=fields["n_steps"], n_invalid=fields["n_invalid"],
                              termination_reason=fields["termination_reason"], input_tokens=tokens[0],
                              output_tokens=tokens[1], seconds=fields["seconds"], answer=fields["answer"],
                              error=fields["error"], extra={"budget": fields["budget"]}))
            with open(episodes_path, "a") as f:
                f.write(json.dumps(episode, default=str) + "\n")
            with open(os.path.join(out_dir, "trajectories", f"{task_id}.json"), "w") as f:
                json.dump(traj, f, indent=1, default=str)
            log_info(f"[android eval] {task_id}: success {fields['success']} ({fields['termination_reason']},"
                     f" {fields['n_steps']} steps)", parameters=parameters)
    try:
        connection.close()
    except Exception:
        pass
    summary = run.summary()
    write_process_episodes(out_dir=out_dir, episodes_path=episodes_path)
    log_info(f"[android eval] {summary} -> {out_dir}", parameters=parameters)


def write_process_episodes(*, out_dir: str, episodes_path: str) -> None:
    """android_world's own metrics table over the latest episode of every task instance."""
    from android_world import suite_utils
    latest = {}
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            latest[(ep["task_template"], ep["instance_id"])] = ep
    if not latest:
        return
    df = suite_utils.process_episodes(list(latest.values()), print_summary=False)
    with open(os.path.join(out_dir, "process_episodes.md"), "w") as f:
        f.write(df.to_markdown())
