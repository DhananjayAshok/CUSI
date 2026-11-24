"""
Practice on the test set itself (GameBoy): the attempt stage of practice (judged tries with critique-hint retries,
one pickle per task, the dataset stage's input), with the tasks the test set's and each try run as the GameBoy
eval runs a task (test mode from its benchmark start state, the eval's step budget, wait ticks and controller). A
diagnostic: it takes the task-proposal leg out of practice, and each try also records the benchmark's own success
(gt_success: the env terminated on the task's goal, as the eval scores it) beside the judge's verdict.

Task ids are t<benchmark row>_r<repeat>; each repeat is an independent attempt of the task (up to max_tries tries).

attempt_split: the same over a split of configs/splits/benchmark_splits.json (cusi.eval.splits) on any env, every
GameBoy game of the split in one run; task ids are <split id>_r<repeat> (gb.pokemon_red.017_r0, web.Apple--3_r0,
aw.ClockTimerEntry_r0). Each try runs as that env's eval runs a task (cusi.eval.gameboy / web / android: test mode,
the eval's budgets and env settings). gt_success is the benchmark's own success where the env has one (GameBoy: the
env terminated on the goal; Android: the task's reward, as the eval scores it) and None on Web, whose eval is a model
judge: each Web try is also written as an eval run (<root>/web_eval/try<k>/), so WebVoyager's auto_eval can score it
afterwards (score_web_tries).

Random prefix (attempt_split random_prefix=n): before the agent's first decision each try takes n random actions
(the env's sample_action) from the benchmark start state, so the agent starts from a perturbed state with the task
known. The actions are drawn from a seed of the task id (which carries the repeat), so every try of a repeat replays
the same prefix and the critique hint applies to the same start; each repeat gets its own. A prefix that ends the
episode (GameBoy: lands on the goal) is redrawn with the next seed, up to PREFIX_REDRAWS times. The env's step budget
is raised by n, so the agent keeps the eval's full budget. The prefix is recorded in each try (entry["prefix"]) and is
not part of the agent's report, so the dataset stage makes no rows from it. Web: test mode is unscoped, so a random
click may leave the task's site (the agent can go back). GameBoy and Web only.
"""
import hashlib
import json
import os
import pickle
import socket
import numpy as np
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
    write_results(paths=paths, parameters=parameters)


def write_results(*, paths: PracticePaths, parameters: dict) -> None:
    """results.csv over every task in the directory, whichever run attempted it."""
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


#: Android tries instantiate each task as the eval does (cusi.eval.android: --task_random_seed 30, instance 0), so a
#: try's goal and parameters are the ones the benchmark scores.
ANDROID_SUITE_SEED = 30
ANDROID_INSTANCE = 0


def split_tasks(*, env_name: str, split: str, repeats: list, parameters: dict, ids: list = None) -> list:
    """Every task of a split on one env (or those of ids), once per repeat, with where its env starts (resolve)."""
    from cusi.eval.gameboy import select_tasks
    from cusi.eval.splits import load_splits, resolve, task_ids
    texts = load_splits(parameters=parameters)
    rows = {}
    out = []
    for r in repeats:
        for sid in task_ids(splits=(split,), env=env_name, parameters=parameters):
            if ids is not None and sid not in ids:
                continue
            where = resolve(task_id=sid, parameters=parameters)
            if env_name == "gameboy":
                game = where["game"]
                if game not in rows:
                    rows[game] = select_tasks(game=game, n_tasks=None)
                where = {**where, "row": rows[game][where["task_index"]]}
                text, scene = where["row"]["task"], f"{game}/{where['row']['init_state']}"
            elif env_name == "web":
                text, scene = where["row"]["ques"], where["row"]["web_name"]
            else:     # the seed-0 text until the first try instantiates the eval's instance (its goal replaces it)
                text, scene = texts[sid]["text"], where["task_class"]
            out.append({"task_id": f"{sid}_r{r}", "split_id": sid, "split": split, "task": text, "scene": scene,
                        "source": split, "node_id": None, "child": None, "repeat": r, "where": where})
    return out


#: Draws of a random prefix that ended the episode before the agent's first decision, before giving up.
PREFIX_REDRAWS = 5


def prefix_seed(*, task_id: str, draw: int) -> int:
    return int.from_bytes(hashlib.sha1(f"prefix:{task_id}:{draw}".encode()).digest()[:8], "big")


def random_prefix(*, env, obs: dict, info: dict, n: int, task_id: str) -> tuple:
    """(obs, info, record) after n random actions from the env's reset state (see the module docstring); record is
    None when n is 0. A draw that ends the episode is redrawn from a fresh reset."""
    if n <= 0:
        return obs, info, None
    for draw in range(PREFIX_REDRAWS):
        if draw:
            obs, info = env.reset()
        seed = prefix_seed(task_id=task_id, draw=draw)
        rng = np.random.default_rng(seed)
        actions, ended = [], False
        for _ in range(n):
            action = env.sample_action(rng=rng)
            obs, _reward, terminated, truncated, info = env.step(action)
            actions.append({"action": action, "valid": info.get("valid"), "error": info.get("error"),
                            "parsed_action": info.get("parsed_action")})
            if terminated or truncated:
                ended = True
                break
        if not ended:
            return obs, info, {"n": n, "draw": draw, "seed": seed, "actions": actions}
    raise RuntimeError(f"random prefix of {n} ended the episode in all {PREFIX_REDRAWS} draws ({task_id})")


def _gameboy_try(*, spec, vlm, task: dict, hint: str, n_try: int, prefix: int, parameters: dict) -> tuple:
    """(result, gt_success, goal, prefix record) of one try, run as the GameBoy eval runs a task."""
    from cusi.envs.gameboy import GameBoyPlayEnv
    from cusi.eval.gameboy import CONTROLLER_VARIANT, GAMEBOY_WAIT_TICKS, MAX_STEPS
    row = task["where"]["row"]
    env = GameBoyPlayEnv(game=row["game"], mode="test", controller_variant=CONTROLLER_VARIANT,
                         max_steps=MAX_STEPS + prefix, wait_ticks=GAMEBOY_WAIT_TICKS, benchmark_row=row,
                         session_name=f"cusi_practice_test/{task['task_id']}_t{n_try}/", parameters=parameters)
    try:
        obs, info = env.reset()
        obs, info, record = random_prefix(env=env, obs=obs, info=info, n=prefix, task_id=task["task_id"])
        agent = spec.make_agent(env=env, vlm=vlm, parameters=parameters)
        with part("agent"):
            _, result = agent.run(task=task["task"], obs=obs, info=info, max_steps=MAX_STEPS, guidance=hint or None)
    finally:
        env.close()
    final = result["report"].last_report
    return result, final is not None and final.termination_reason == "terminated", task["task"], record


def _web_try(*, spec, vlm, task: dict, hint: str, n_try: int, root: str, prefix: int, parameters: dict) -> tuple:
    """(result, None, goal, prefix record) of one try, run as the Web eval runs a task (its browser-death retries
    included), and written as task <id> of the eval run <root>/web_eval/try<n_try>/ for auto_eval."""
    import shutil
    import time
    from cusi.envs.webvoyager import WebVoyagerPlayEnv, filtered_task_file, load_webvoyager
    from cusi.eval import web as web_eval
    from cusi.utils.paths import web_task_dir
    wv = load_webvoyager(project_root=parameters["project_root"])
    row = task["where"]["row"]
    run_dir = os.path.join(root, "web_eval", f"try{n_try}")
    task_dir = web_task_dir(run_dir=run_dir, task_id=row["id"])
    t0 = time.time()
    for attempt in range(1, web_eval.MAX_TASK_ATTEMPTS + 1):
        shutil.rmtree(task_dir, ignore_errors=True)
        os.makedirs(task_dir)
        env, supervisor, result, crashed, error, record = None, None, None, None, None, None
        try:
            env = WebVoyagerPlayEnv(task_id=row["id"], mode="test", data_file=filtered_task_file(parameters=parameters),
                                    max_steps=web_eval.MAX_ITER + prefix, fast_waits=False, headless=True,
                                    fix_box_color=True, window_width=1024, window_height=768, parameters=parameters)
            obs, info = env.reset()
            obs, info, record = random_prefix(env=env, obs=obs, info=info, n=prefix, task_id=task["task_id"])
            agent = spec.make_agent(env=env, vlm=vlm, parameters=parameters)
            with part("agent"):
                supervisor, result = agent.run(task=row["ques"], obs=obs, info=info, max_steps=web_eval.MAX_ITER,
                                               guidance=hint or None)
            if any(s.error == web_eval.BROWSER_CRASHED for r in result["report"].executor_reports
                   for s in r.env_steps):
                crashed = "browser died mid-task"
        except wv.WebDriverException as e:
            if wv._is_browser_dead(e):
                crashed = (str(e).splitlines() or [type(e).__name__])[0][:200]
            else:
                error = f"driver error: {str(e)[:200]}"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        if crashed is None:
            break
        time.sleep(5 * attempt)
    if result is None:
        raise RuntimeError(f"web try {task['task_id']} t{n_try} did not run: {error or crashed}")
    report = result["report"]
    screenshots = web_eval.write_artefacts(task_dir=task_dir, executors=supervisor.executors,
                                           system_prompt=wv.SYSTEM_PROMPT, task=row["ques"])
    final = report.last_report
    web_eval._finish(task=row, task_dir=task_dir, run_dir=run_dir, report=report, result=result,
                     screenshots=screenshots, error=error or (final.error if final else None),
                     model_name=vlm.model_name, seconds=time.time() - t0, compute={})
    return result, None, row["ques"], record


class AndroidTries:
    """One emulator's tries (a process attempts Android tasks one at a time on it, as the eval does)."""

    def __init__(self, *, console_port: int, grpc_port: int, parameters: dict) -> None:
        import shutil
        from android_world.env import env_launcher
        from cusi.envs.android_world import AndroidEmulator
        self.console_port, self.grpc_port, self._parameters = console_port, grpc_port, parameters
        self.emulator = AndroidEmulator(port=console_port, grpc_port=grpc_port, snapshots=False, parameters=parameters)
        self.connection = env_launcher.load_and_setup_env(console_port=console_port, grpc_port=grpc_port,
                                                          emulator_setup=False, adb_path=shutil.which("adb"))

    def __call__(self, *, spec, vlm, task: dict, hint: str, n_try: int) -> tuple:
        """(result, gt_success, goal): gt_success is the task's reward (as the eval scores it: 0 unless the agent
        ended the task), None when the emulator hung mid-try."""
        from cusi.envs.android_world import AndroidPlayEnv
        env = None
        try:
            env = AndroidPlayEnv(task=task["where"]["task_class"], mode="test", suite_seed=ANDROID_SUITE_SEED,
                                 instance=ANDROID_INSTANCE, console_port=self.console_port, grpc_port=self.grpc_port,
                                 emulator=self.emulator, reset_mode="reinit", connection=self.connection,
                                 stabilize_after_action=False, wait_after_action_seconds=2.0,
                                 parameters=self._parameters)
            goal, recoveries = env._task.goal, env.recoveries
            obs, info = env.reset()
            agent = spec.make_agent(env=env, vlm=vlm, parameters=self._parameters)
            with part("agent"):
                _, result = agent.run(task=goal, obs=obs, info=info, max_steps=env.max_steps, guidance=hint or None)
            final = result["report"].last_report
            if env.recoveries > recoveries:
                gt = None
            else:
                done = final is not None and final.termination_reason == "terminated"
                gt = done and float(final.final_reward or 0.0) >= 1.0
                env._guarded(fn=lambda: env._task.tear_down(env._env))
        finally:
            if env is not None:
                self.connection = env._env
                env.close(close_connection=False)
        return result, gt, goal, None

    def close(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass


def attempt_split(*, spec, vlm, paths: PracticePaths, split: str, repeats: list, max_tries: int, lookback: int,
                  judge_max_new_tokens: int, n_workers: int, parameters: dict, ids: list = None, shard: str = None,
                  console_port: int = 5554, grpc_port: int = 8554, random_prefix_n: int = 0, judge_vlm=None) -> None:
    """Practice on a split's tasks of spec's env, each try run as the env's eval runs the task, judged with
    critique-hint retries by judge_vlm (default: vlm itself). shard "k/n": every n-th task from the k-th (processes
    sharing paths). Android runs its tasks one at a time on the emulator at console_port / grpc_port (n_workers is
    ignored). random_prefix_n: random actions before the agent's first decision (module docstring)."""
    from cusi.agents.specs import gameboy_spec
    judge_vlm = judge_vlm or vlm
    if random_prefix_n and spec.name == "android":
        log_error("attempt_split: the random prefix is GameBoy and Web only", parameters=parameters)
    tasks = split_tasks(env_name=spec.name, split=split, repeats=repeats, parameters=parameters, ids=ids)
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        tasks = tasks[k::n]
    os.makedirs(paths.episodes_dir, exist_ok=True)
    # shards write this at once: each through its own temporary file
    tmp = os.path.join(paths.root, f"config.json.{socket.gethostname()}.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump({"env": spec.name, "model_name": vlm.model_name, "agent": AGENT_VERSION, "max_tries": max_tries,
                   "lookback": lookback, "tasks": "split", "split": split, "random_prefix": random_prefix_n,
                   "temperature": vlm.temperature, "judge_greedy": judge_vlm.greedy}, f, indent=2)
    os.replace(tmp, os.path.join(paths.root, "config.json"))
    jobs = [t for t in tasks if not os.path.exists(os.path.join(paths.episodes_dir, f"{t['task_id']}.pkl"))]
    log_info(f"attempt_split {spec.name} {split}{f' shard {shard}' if shard else ''}: {len(tasks) - len(jobs)} of "
             f"{len(tasks)} tasks done, {len(jobs)} to run.", parameters=parameters)
    android = AndroidTries(console_port=console_port, grpc_port=grpc_port, parameters=parameters) \
        if spec.name == "android" and jobs else None

    def run_try(task: dict, hint: str, n_try: int) -> tuple:
        if spec.name == "gameboy":
            return _gameboy_try(spec=spec, vlm=vlm, task=task, hint=hint, n_try=n_try, prefix=random_prefix_n,
                                parameters=parameters)
        if spec.name == "web":
            return _web_try(spec=spec, vlm=vlm, task=task, hint=hint, n_try=n_try, root=paths.root,
                            prefix=random_prefix_n, parameters=parameters)
        return android(spec=spec, vlm=vlm, task=task, hint=hint, n_try=n_try)

    def one(task: dict, *, worker: int) -> dict:
        # the judge and critique read the game's own domain (GameBoy runs every game of the split)
        domain = gameboy_spec(game=task["where"]["game"]).domain if spec.name == "gameboy" else spec.domain
        record = {"task": {k: v for k, v in task.items() if k != "where"}, "success": False, "restore_error": None,
                  "tries": []}
        hint = ""
        with tally() as task_compute:
            for n_try in range(1, max_tries + 1):
                with tally() as try_compute:
                    result, gt, goal, prefix = run_try(task, hint, n_try)
                    record["task"]["task"] = goal     # Android: the eval instance's goal, not the seed-0 text
                    leg = merged_leg(report=result["report"])
                    with part("judge"):
                        verdict = judge_leg(report=leg, vlm=judge_vlm, domain=domain, lookback=lookback,
                                            max_new_tokens=judge_max_new_tokens)
                    entry = {"try": n_try, "hint": hint or None, "report": result["report"],
                             "supervisor_state": supervisor_state(result=result), "verdict": verdict,
                             "gt_success": gt, "critique_calls": [], "prefix": prefix}
                    record["tries"].append(entry)
                    log_info(f"attempt_split [{task['task_id']}] try {n_try}: judge "
                             f"{'success' if verdict['success'] else 'failure'}, benchmark "
                             f"{'n/a' if gt is None else 'success' if gt else 'failure'} ({leg.termination_reason}, "
                             f"{len(leg.env_steps)} steps) | {goal}", parameters=parameters)
                    if not verdict["success"] and n_try < max_tries:
                        with part("critique"):
                            hint = derive_critique_hint(report=leg, vlm=judge_vlm, domain=domain, previous_hint=hint,
                                                        log=entry["critique_calls"],
                                                        max_new_tokens=judge_max_new_tokens)
                entry["compute"] = try_compute.snapshot()
                if verdict["success"]:
                    record["success"] = True
                    break
        record["compute"] = task_compute.snapshot()
        atomic_pickle(record, os.path.join(paths.episodes_dir, f"{task['task_id']}.pkl"))
        return record

    workers = 1 if spec.name == "android" else min(n_workers, max(1, len(jobs)))
    try:
        run_jobs(jobs=jobs, fn=one, n_workers=workers, desc="attempt_split", on_result=lambda job, record: None,
                 parameters=parameters)
    finally:
        if android is not None:
            android.close()
    write_results(paths=paths, parameters=parameters)


def score_web_tries(*, paths: PracticePaths, judge_model_name: str, judge_model_backend: str,
                    judge_vllm_base_url, parameters: dict) -> dict:
    """WebVoyager's auto_eval (the Web benchmark's own judge) on every Web try written by attempt_split
    (<root>/web_eval/try<k>/): {"try<k>": {task id: "success" | "fail" | ...}}, also saved as <root>/web_scores.json."""
    from cusi.eval import web as web_eval
    root = os.path.join(paths.root, "web_eval")
    out = {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        scores = web_eval.judge(run_dir=os.path.join(root, name), judge_model_name=judge_model_name,
                                judge_model_backend=judge_model_backend, judge_vllm_base_url=judge_vllm_base_url,
                                parameters=parameters)
        out[name] = scores["tasks"]
        log_info(f"score_web_tries {name}: {scores['success']}/{scores['scored']} by {judge_model_name}",
                 parameters=parameters)
    out_path = os.path.join(paths.root, "web_scores.json")
    with open(out_path + ".tmp", "w") as f:
        json.dump({"judge_model_name": judge_model_name, "tries": out}, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    return out


def load_web_scores(*, root: str) -> dict:
    """{"try<k>": {WebVoyager id: verdict}} from <root>/web_scores.json (score_web_tries), {} if not scored."""
    path = os.path.join(root, "web_scores.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)["tries"]


def try_gt(*, record: dict, entry: dict, web_scores: dict):
    """A try's benchmark success: gt_success, or on Web the try's auto_eval verdict (keyed by WebVoyager id); None
    if neither exists."""
    gt = entry["gt_success"]
    if gt is None and web_scores:
        verdict = web_scores.get(f"try{entry['try']}", {}).get(record["task"]["split_id"][len("web."):])
        gt = None if verdict is None else verdict == "success"
    return gt


def judge_agreement(*, paths: PracticePaths, parameters: dict) -> dict:
    """Over every try in the directory: the practice judge's verdicts against the benchmark's success (gt_success;
    on Web, auto_eval's verdict from web_scores.json where present), as counts, precision and recall, plus the
    tasks judged successful and those the benchmark scores successful at some try. Saved as judge_agreement.json."""
    web_scores = load_web_scores(root=paths.root)
    counts = {"TT": 0, "TF": 0, "FT": 0, "FF": 0, "judge_only": 0}
    tasks = {"n": 0, "judged_success": 0, "gt_success": 0}
    by_try = {}
    for name in sorted(os.listdir(paths.episodes_dir)):
        if not name.endswith(".pkl"):
            continue
        with open(os.path.join(paths.episodes_dir, name), "rb") as f:
            record = pickle.load(f)
        tasks["n"] += 1
        tasks["judged_success"] += record["success"]
        task_gt = False
        for entry in record["tries"]:
            gt = try_gt(record=record, entry=entry, web_scores=web_scores)
            judge = bool(entry["verdict"]["success"])
            if gt is None:
                counts["judge_only"] += 1
                continue
            task_gt |= gt
            counts[("T" if judge else "F") + ("T" if gt else "F")] += 1
            t = by_try.setdefault(entry["try"], {"TT": 0, "TF": 0, "FT": 0, "FF": 0})
            t[("T" if judge else "F") + ("T" if gt else "F")] += 1
        tasks["gt_success"] += task_gt
    precision = counts["TT"] / max(1, counts["TT"] + counts["TF"])
    recall = counts["TT"] / max(1, counts["TT"] + counts["FT"])
    out = {"counts": counts, "precision": precision, "recall": recall, "tasks": tasks, "by_try": by_try}
    with open(os.path.join(paths.root, "judge_agreement.json"), "w") as f:
        json.dump(out, f, indent=1)
    log_info(f"judge_agreement {paths.root}: {out}", parameters=parameters)
    return out
