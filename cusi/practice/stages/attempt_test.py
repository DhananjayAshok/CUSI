"""
Practice on the test set itself (GameBoy): the attempt stage of practice (judged tries with critique-hint retries,
one pickle per task, the dataset stage's input), with the tasks the test set's and each try run as the GameBoy
eval runs a task (test mode from its benchmark start state, the eval's step budget, wait ticks and controller). A
diagnostic: it takes the task-proposal leg out of practice, and each try also records the benchmark's own success
(gt_success: the env terminated on the task's goal, as the eval scores it) beside the judge's verdict.

Task ids are t<benchmark row>_r<repeat>; each repeat is an independent attempt of the task (up to max_tries tries).
"""
import json
import os
import pickle
import socket
import pandas as pd
from cusi.agents.agent import AGENT_VERSION, merged_leg
from cusi.eval.supervision import supervisor_state
from cusi.practice.judging import derive_critique_hint, judge_leg
from cusi.practice.stages.common import atomic_pickle, run_jobs
from cusi.utils.compute import part, tally
from cusi.utils.log_handling import log_error, log_info
from cusi.utils.paths import PracticePaths


def test_tasks(*, game: str, repeats: list, indices: list = None) -> list:
    """Every benchmark task of game (or those at indices), once per repeat, with the benchmark row it runs from."""
    from cusi.eval.gameboy import select_tasks
    return [{"task_id": f"t{i:03d}_r{r}", "task": row["task"], "scene": row["init_state"], "source": "test",
             "node_id": None, "child": None, "benchmark_index": i, "repeat": r, "row": row}
            for r in repeats for i, row in enumerate(select_tasks(game=game, n_tasks=None))
            if indices is None or i in indices]


def attempt_test(*, spec, vlm, paths: PracticePaths, game: str, repeats: list, max_tries: int, lookback: int,
                 judge_max_new_tokens: int, n_workers: int, parameters: dict, indices: list = None) -> None:
    """repeats: which repeats this run attempts (jobs splitting the repeats share paths); indices: only these
    benchmark rows (default all)."""
    from cusi.envs.gameboy import GameBoyPlayEnv
    from cusi.eval.gameboy import CONTROLLER_VARIANT, GAMEBOY_WAIT_TICKS, MAX_STEPS
    if spec.name != "gameboy":
        log_error(f"attempt_test: only GameBoy has test-set practice, not {spec.name}", parameters=parameters)
    tasks = test_tasks(game=game, repeats=repeats, indices=indices)
    os.makedirs(paths.episodes_dir, exist_ok=True)
    # jobs splitting the repeats write this at once: each through its own temporary file
    tmp = os.path.join(paths.root, f"config.json.{socket.gethostname()}.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump({"env": spec.name, "model_name": vlm.model_name, "agent": AGENT_VERSION, "max_tries": max_tries,
                   "max_steps": MAX_STEPS, "lookback": lookback, "tasks": "test", "game": game}, f, indent=2)
    os.replace(tmp, os.path.join(paths.root, "config.json"))
    jobs = [t for t in tasks if not os.path.exists(os.path.join(paths.episodes_dir, f"{t['task_id']}.pkl"))]
    log_info(f"attempt_test: {len(tasks) - len(jobs)} of {len(tasks)} tasks done, {len(jobs)} to run.",
             parameters=parameters)

    def one(task: dict, *, worker: int) -> dict:
        row = task["row"]
        record = {"task": {k: v for k, v in task.items() if k != "row"}, "success": False, "restore_error": None,
                  "tries": []}
        hint = ""
        with tally() as task_compute:
            for n_try in range(1, max_tries + 1):
                with tally() as try_compute:
                    env = GameBoyPlayEnv(game=game, mode="test", controller_variant=CONTROLLER_VARIANT,
                                         max_steps=MAX_STEPS, wait_ticks=GAMEBOY_WAIT_TICKS, benchmark_row=row,
                                         session_name=f"cusi_practice_test/{task['task_id']}_t{n_try}/",
                                         parameters=parameters)
                    try:
                        obs, info = env.reset()
                        agent = spec.make_agent(env=env, vlm=vlm, parameters=parameters)
                        with part("agent"):
                            _, result = agent.run(task=task["task"], obs=obs, info=info, max_steps=MAX_STEPS,
                                                  guidance=hint or None)
                    finally:
                        env.close()
                    final = result["report"].last_report
                    gt = final is not None and final.termination_reason == "terminated"   # as the eval scores it
                    leg = merged_leg(report=result["report"])
                    with part("judge"):
                        verdict = judge_leg(report=leg, vlm=vlm, domain=spec.domain, lookback=lookback,
                                            max_new_tokens=judge_max_new_tokens)
                    entry = {"try": n_try, "hint": hint or None, "report": result["report"],
                             "supervisor_state": supervisor_state(result=result), "verdict": verdict,
                             "gt_success": gt, "critique_calls": []}
                    record["tries"].append(entry)
                    log_info(f"attempt_test [{task['task_id']}] try {n_try}: judge "
                             f"{'success' if verdict['success'] else 'failure'}, benchmark "
                             f"{'success' if gt else 'failure'} ({leg.termination_reason}, {len(leg.env_steps)} "
                             f"steps) | {task['task']}", parameters=parameters)
                    if not verdict["success"] and n_try < max_tries:
                        with part("critique"):
                            hint = derive_critique_hint(report=leg, vlm=vlm, domain=spec.domain, previous_hint=hint,
                                                        log=entry["critique_calls"],
                                                        max_new_tokens=judge_max_new_tokens)
                entry["compute"] = try_compute.snapshot()
                if verdict["success"]:
                    record["success"] = True
                    break
        record["compute"] = task_compute.snapshot()
        atomic_pickle(record, os.path.join(paths.episodes_dir, f"{task['task_id']}.pkl"))
        return record

    run_jobs(jobs=jobs, fn=one, n_workers=min(n_workers, max(1, len(jobs))), desc="attempt_test",
             on_result=lambda job, record: None, parameters=parameters)
    # results.csv covers every task in the directory, whichever run attempted it
    rows = []
    for name in sorted(os.listdir(paths.episodes_dir)):
        if not name.endswith(".pkl"):
            continue
        with open(os.path.join(paths.episodes_dir, name), "rb") as f:
            record = pickle.load(f)
        tries = record["tries"]
        rows.append({**record["task"], "success": record["success"], "n_tries": len(tries),
                     "judge": [t["verdict"]["success"] for t in tries], "gt": [t["gt_success"] for t in tries],
                     "n_env_steps": [t["verdict"]["n_env_steps"] for t in tries],
                     "judge_reasoning": tries[-1]["verdict"]["reasoning"] if tries else ""})
    pd.DataFrame(rows).to_csv(paths.results, index=False)
    log_info(f"attempt_test: {sum(r['success'] for r in rows)}/{len(rows)} tasks judged successful, "
             f"{sum(any(r['gt']) for r in rows)} with a benchmark success -> {paths.root}", parameters=parameters)
