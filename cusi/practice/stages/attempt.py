"""
Stage 2: attempt each proposed task, judged, retrying with a critique hint on failure.
"""
import json
import os
import pickle
import pandas as pd
from cusi.utils.log_handling import log_info, log_warn
from cusi.practice.judging import derive_critique_hint, judge_leg
from cusi.practice.stages.common import atomic_json, atomic_pickle, load_json, run_jobs
from cusi.utils.paths import PracticePaths


def attempt(*, spec, pool, vlm, paths: PracticePaths, max_attempts: int, lookback: int, judge_max_new_tokens: int,
            executor_kwargs: dict, overwrite: bool, parameters: dict) -> None:
    out_dir = paths.attempts_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    if os.path.exists(csv_path) and not overwrite:
        log_info(f"attempt: already done ({csv_path}).", parameters=parameters)
        return
    ckpt_json, ckpt_pkl = os.path.join(out_dir, "checkpoint.json"), os.path.join(out_dir, "checkpoint.pkl")
    results = {} if overwrite else load_json(ckpt_json, {})
    legs = {}
    if not overwrite and os.path.exists(ckpt_pkl):
        with open(ckpt_pkl, "rb") as f:
            legs = pickle.load(f)
    results = {g: r for g, r in results.items() if g in legs}   # a result counts only with its leg

    with open(paths.proposals) as f:
        proposals = [json.loads(line) for line in f]
    jobs = [(f"{p['scene']}_{i}", p["scene"], task) for p in proposals for i, task in enumerate(p["tasks"])
            if f"{p['scene']}_{i}" not in results]
    log_info(f"attempt: {len(results)} done, {len(jobs)} to run.", parameters=parameters)

    def one(job, *, worker: int):
        group, scene, task = job
        env = pool.get(worker=worker, scene=scene)
        executor = spec.make_executor(env=env, vlm=vlm, parameters=parameters, **executor_kwargs)
        hint, verdict, report = "", None, None
        for n_try in range(1, max_attempts + 1):
            obs, info = env.reset()
            report = executor.run(task=task, hint=hint or None, max_steps=spec.max_steps, obs=obs, info=info,
                                  allow_done_check=spec.name == "gameboy", env_name=spec.name)
            verdict = judge_leg(report=report, vlm=vlm, domain=spec.domain, lookback=lookback,
                                max_new_tokens=judge_max_new_tokens)
            log_info(f"attempt [{group}] try {n_try}: {'success' if verdict['success'] else 'failure'} "
                     f"({report.termination_reason}, {len(report.env_steps)} steps) | {task}", parameters=parameters)
            if verdict["success"]:
                break
            if n_try < max_attempts:
                hint = derive_critique_hint(report=report, vlm=vlm, domain=spec.domain, previous_hint=hint,
                                            max_new_tokens=judge_max_new_tokens)
        row = {"group_idx": group, "scene": scene, "task_string": task, "success": verdict["success"],
               "n_tries": n_try, "final_hint": hint, "judge_description": verdict["description"],
               "judge_reasoning": verdict["reasoning"], "safe_success_point": verdict["safe_success_point"],
               "termination_reason": verdict["termination_reason"], "n_env_steps": verdict["n_env_steps"],
               "max_steps": verdict["max_steps"], "answer": report.answer,
               "input_tokens": sum((c.input_tokens or 0) for c in report.calls),
               "output_tokens": sum((c.output_tokens or 0) for c in report.calls)}
        return row, report

    def save(job, result):
        row, report = result
        results[row["group_idx"]] = row
        legs[row["group_idx"]] = report
        atomic_pickle(legs, ckpt_pkl)
        atomic_json(results, ckpt_json)

    failures = run_jobs(jobs=jobs, fn=one, n_workers=min(spec.max_workers, max(1, len(jobs))), desc="attempt",
                        on_result=save, parameters=parameters)
    if failures:
        log_warn("attempt: some tasks failed with an exception; not finalising. Rerun to retry them.",
                 parameters=parameters)
        return
    order = [g for p in proposals for g in (f"{p['scene']}_{i}" for i in range(len(p["tasks"])))]
    pd.DataFrame([results[g] for g in order if g in results]).to_csv(csv_path, index=False)
    atomic_pickle(legs, os.path.join(out_dir, "legs.pkl"))
    success = {g: {"task": results[g]["task_string"], "scene": results[g]["scene"]}
               for g in order if g in results and results[g]["success"]}
    atomic_json(success, os.path.join(out_dir, "success.json"))
    log_info(f"attempt: {len(success)}/{len(results)} tasks succeeded -> {out_dir}", parameters=parameters)
    for p in (ckpt_json, ckpt_pkl):
        if os.path.exists(p):
            os.remove(p)
