"""Stage 4: practise each successful task (GameBoyRL practice_tasks).

Each (group, attempt) episode: reset with a deterministic seed, perturb the start with
n_random_actions random actions (env.sample_action), run the executor with the guidance as its
hint and the goal condition for the judge only; on failure derive a critique hint, replay the
same perturbation and retry once with "guidance + Specific hint". Every executor call (prompt,
images, response) is pickled per episode as the LegReport of the final try.

Output in practice/: <group>_<attempt>.pkl, results.csv, config.json (checkpoint.json resumes).
"""
import os
import pandas as pd
from cusi_utils.log_handling import log_info, log_warn
from cusi_practice.judging import derive_critique_hint, format_guidance, judge_leg
from cusi_practice.stages.common import PracticePaths, atomic_json, atomic_pickle, load_json, run_jobs


def episode_seed(*, base_seed: int, group_index: int, attempt: int) -> int:
    """_episode_seed's layout (decimal place values), on the group's position."""
    return base_seed + group_index * 100 + attempt


def perturb(*, env, seed: int, n_random_actions: int) -> tuple:
    """reset(seed) then n random actions. Returns (obs, info, actions)."""
    obs, info = env.reset(seed=seed)
    actions = []
    for _ in range(n_random_actions):
        action = env.sample_action()
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=seed)
            actions = []
    return obs, info, actions


def replay(*, env, seed: int, actions: list) -> tuple:
    obs, info = env.reset(seed=seed)
    for action in actions:
        obs, _, _, _, info = env.step(action)
    return obs, info


def practice(*, spec, pool, vlm, paths: PracticePaths, n_attempts: int, n_random_actions: int, lookback: int,
             judge_max_new_tokens: int, executor_kwargs: dict, overwrite: bool, parameters: dict) -> None:
    out_dir = paths.practice_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    if os.path.exists(csv_path) and not overwrite:
        log_info(f"practice: already done ({csv_path}).", parameters=parameters)
        return
    guidance_data = load_json(paths.guidance, None)
    if guidance_data is None:
        log_warn("practice: no guidance.json; run the guidance stage first.", parameters=parameters)
        return
    if not guidance_data:
        log_warn(f"practice: no tasks with guidance for {spec.name}; nothing to practise.", parameters=parameters)
        pd.DataFrame([]).to_csv(csv_path, index=False)
        return
    base_seed = parameters["random_seed"]
    ckpt = os.path.join(out_dir, "checkpoint.json")
    rows = [] if overwrite else load_json(ckpt, [])
    done = {(r["group_idx"], r["attempt"]) for r in rows}
    groups = list(guidance_data)
    # Grouped by scene: switching scene rebuilds the environment (about 30 s on Android).
    jobs = sorted([(g, a) for g in groups for a in range(n_attempts) if (g, a) not in done],
                  key=lambda j: (guidance_data[j[0]]["scene"], j[0], j[1]))
    atomic_json({"env": spec.name, "model_name": vlm.model_name, "n_tasks": len(groups), "n_attempts": n_attempts,
                 "n_random_actions": n_random_actions, "max_steps": spec.max_steps, "lookback": lookback,
                 "base_seed": base_seed, "executor_kwargs": executor_kwargs}, os.path.join(out_dir, "config.json"))
    log_info(f"practice: {len(done)} episodes done, {len(jobs)} to run.", parameters=parameters)

    def one(job, *, worker: int):
        group, attempt = job
        record = guidance_data[group]
        task, scene = record["task"], record["scene"]
        goal_condition = record.get("goal_condition") or None
        guidance_str = format_guidance(record.get("guidance", {}))
        env = pool.get(worker=worker, scene=scene)
        executor = spec.make_executor(env=env, vlm=vlm, parameters=parameters, **executor_kwargs)
        seed = episode_seed(base_seed=base_seed, group_index=groups.index(group), attempt=attempt)
        obs, info, actions = perturb(env=env, seed=seed, n_random_actions=n_random_actions)
        report = executor.run(task=task, hint=guidance_str or None, max_steps=spec.max_steps, obs=obs, info=info,
                              env_name=spec.name)
        verdict = judge_leg(report=report, vlm=vlm, domain=spec.domain, lookback=lookback,
                            goal_condition=goal_condition, max_new_tokens=judge_max_new_tokens)
        failed = not verdict["success"]
        derived_hint = ""
        if failed:
            derived_hint = derive_critique_hint(report=report, vlm=vlm, domain=spec.domain,
                                                max_new_tokens=judge_max_new_tokens)
            hint = f"{guidance_str}\nSpecific hint: {derived_hint}" if guidance_str else f"Specific hint: {derived_hint}"
            obs, info = replay(env=env, seed=seed, actions=actions)
            report = executor.run(task=task, hint=hint, max_steps=spec.max_steps, obs=obs, info=info,
                                  env_name=spec.name)
            verdict = judge_leg(report=report, vlm=vlm, domain=spec.domain, lookback=lookback,
                                goal_condition=goal_condition, max_new_tokens=judge_max_new_tokens)
        log_info(f"practice [{group}_{attempt}]: {'success' if verdict['success'] else 'failure'}"
                 f"{' (after retry)' if failed else ''} | {task}", parameters=parameters)
        row = {"group_idx": group, "attempt": attempt, "scene": scene, "task_string": task,
               "success": verdict["success"], "safe_success_point": verdict["safe_success_point"], "seed": seed,
               "random_actions": actions, "used_retry": failed, "derived_hint": derived_hint,
               "guidance": guidance_str, "goal_condition": goal_condition or "",
               "judge_description": verdict["description"], "judge_reasoning": verdict["reasoning"],
               "termination_reason": verdict["termination_reason"], "n_env_steps": verdict["n_env_steps"],
               "answer": report.answer}
        return row, report

    def save(job, result):
        row, report = result
        atomic_pickle(report, os.path.join(out_dir, f"{row['group_idx']}_{row['attempt']}.pkl"))
        rows.append(row)
        atomic_json(rows, ckpt)

    failures = run_jobs(jobs=jobs, fn=one, n_workers=min(spec.max_workers, max(1, len(jobs))), desc="practice",
                        on_result=save, parameters=parameters)
    if failures:
        log_warn("practice: some episodes failed with an exception; not finalising. Rerun to retry them.",
                 parameters=parameters)
        return
    order = {g: i for i, g in enumerate(groups)}
    rows.sort(key=lambda r: (order[r["group_idx"]], r["attempt"]))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    n_success = sum(bool(r["success"]) for r in rows)
    log_info(f"practice: {n_success}/{len(rows)} episodes succeeded -> {out_dir}", parameters=parameters)
