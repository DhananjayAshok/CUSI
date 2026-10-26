"""
Attempt each task with the env's final agent (executor under its supervisor), from its start: the tree node it was
inferred at, restored from an annotated search run. The same model
judges each try (judge_leg); after a failure it writes a critique hint (derive_critique_hint), shown to the executor
alone as guidance on the next try from the same start, up to max_tries.

Each task's tries are saved as one pickle (episodes/<task_id>.pkl), the input of the dataset stage:
    {"task": {task_id, task, scene, source, node_id, child}, "success", "restore_error",
     "tries": [{"try", "hint", "report" (SupervisorReport: supervisor calls and every executor leg with its calls and
                frames), "supervisor_state", "verdict" (judge_leg's, with the judge's calls),
                "critique_calls"}]}
results.csv has one row per task. A task with a pickle is done; rerunning continues with the rest.
"""
import os
import pickle
import pandas as pd
from cusi.agents.agent import AGENT_VERSION, merged_leg
from cusi.eval.supervision import supervisor_state
from cusi.practice.judging import derive_critique_hint, judge_leg
from cusi.practice.stages.common import atomic_json, atomic_pickle, run_jobs
from cusi.search.restore import NodeRestorer, RestoreFailed
from cusi.utils.compute import part, tally
from cusi.utils.log_handling import log_error, log_info, log_warn
from cusi.utils.paths import PracticePaths


def tree_tasks(*, restorer: NodeRestorer, parameters: dict) -> list:
    """Every task of an annotated search run's nodes, in node order (unrestorable nodes' tasks are left out)."""
    nodes = restorer.nodes
    if not any(n.get("node_notes_raw") is not None for n in nodes.values()):
        log_error(f"{restorer.directory} has no task annotations: run cusi.search.annotate first",
                  parameters=parameters)
    out = []
    for nid in sorted(nodes):
        flags = nodes[nid].get("flags") or {}
        if flags.get("unrestorable") or flags.get("replay_diverged") or flags.get("challenge_page"):
            continue
        for i, t in enumerate(nodes[nid].get("tasks") or []):
            out.append({"task_id": f"n{nid}_{i}", "task": t["task"], "scene": nodes[nid]["env_key"],
                        "source": t["source"], "node_id": nid, "child": t.get("child")})
    return out


def compute_columns(compute) -> dict:
    """A task's compute as results.csv columns (empty for a task attempted before compute was recorded)."""
    if not compute:
        return {}
    llm = compute["llm"]
    return {"wall_s": round(compute["wall_s"]["total"], 1), "llm_calls": llm["calls"],
            "llm_prompt_tokens": llm["prompt_tokens"], "llm_completion_tokens": llm["completion_tokens"],
            "llm_flops": llm["flops"], "local_gpu_s": round(sum(g["s"] for g in compute["gpu"].values()), 2),
            "local_gpu_flops": sum(g["flops"] for g in compute["gpu"].values())}


def attempt(*, spec, pool, vlm, paths: PracticePaths, restorer: NodeRestorer, max_tries: int, lookback: int,
            judge_max_new_tokens: int, max_tasks, parameters: dict) -> None:
    """restorer: the annotated search run's NodeRestorer. max_tasks: the first this many (None: all)."""
    tasks = tree_tasks(restorer=restorer, parameters=parameters)
    tasks = tasks[:max_tasks] if max_tasks is not None else tasks
    os.makedirs(paths.episodes_dir, exist_ok=True)
    atomic_json({"env": spec.name, "model_name": vlm.model_name, "agent": AGENT_VERSION, "max_tries": max_tries,
                 "max_steps": spec.max_steps, "lookback": lookback,
                 "tree": restorer.directory},
                os.path.join(paths.root, "config.json"))
    done = {t["task_id"] for t in tasks if os.path.exists(os.path.join(paths.episodes_dir, f"{t['task_id']}.pkl"))}
    jobs = [t for t in tasks if t["task_id"] not in done]
    # Grouped by scene: a worker switching scene rebuilds its env.
    jobs.sort(key=lambda t: t["scene"])
    log_info(f"attempt: {len(done)} of {len(tasks)} tasks done, {len(jobs)} to run.", parameters=parameters)

    def start(env, task: dict) -> tuple:
        return restorer.restore(env=env, node_id=task["node_id"])

    def one(task: dict, *, worker: int) -> dict:
        env = pool.get(worker=worker, scene=task["scene"])
        agent = spec.make_agent(env=env, vlm=vlm, parameters=parameters)
        record = {"task": task, "success": False, "restore_error": None, "tries": []}
        hint = ""
        with tally() as task_compute:
            for n_try in range(1, max_tries + 1):
                with tally() as try_compute:
                    try:
                        with part("restore"):
                            obs, info = start(env, task)
                    except RestoreFailed as e:
                        record["restore_error"] = str(e)
                        log_warn(f"attempt [{task['task_id']}]: {e}", parameters=parameters)
                        break
                    with part("agent"):
                        _, result = agent.run(task=task["task"], obs=obs, info=info, max_steps=spec.max_steps,
                                              guidance=hint or None)
                    leg = merged_leg(report=result["report"])
                    with part("judge"):
                        verdict = judge_leg(report=leg, vlm=vlm, domain=spec.domain, lookback=lookback,
                                            max_new_tokens=judge_max_new_tokens)
                    entry = {"try": n_try, "hint": hint or None, "report": result["report"],
                             "supervisor_state": supervisor_state(result=result), "verdict": verdict,
                             "critique_calls": []}
                    record["tries"].append(entry)
                    log_info(f"attempt [{task['task_id']}] try {n_try}: "
                             f"{'success' if verdict['success'] else 'failure'} ({leg.termination_reason}, "
                             f"{len(leg.env_steps)} steps) | {task['task']}", parameters=parameters)
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

    failures = run_jobs(jobs=jobs, fn=one, n_workers=min(spec.max_workers, max(1, len(jobs))), desc="attempt",
                        on_result=lambda job, record: None, parameters=parameters)
    rows = []
    for task in tasks:
        path = os.path.join(paths.episodes_dir, f"{task['task_id']}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            record = pickle.load(f)
        tries = record["tries"]
        rows.append({**task, "success": record["success"], "n_tries": len(tries),
                     "restore_error": record["restore_error"],
                     "termination_reasons": [t["verdict"]["termination_reason"] for t in tries],
                     "n_env_steps": [t["verdict"]["n_env_steps"] for t in tries],
                     "judge_reasoning": tries[-1]["verdict"]["reasoning"] if tries else "",
                     **compute_columns(record.get("compute"))})
    pd.DataFrame(rows).to_csv(paths.results, index=False)
    n_ok = sum(r["success"] for r in rows)
    log_info(f"attempt: {n_ok}/{len(rows)} tasks succeeded ({sum(r['n_tries'] > 1 and r['success'] for r in rows)} "
             f"after a hint) -> {paths.root}", parameters=parameters)
    if failures:
        log_warn("attempt: some tasks raised; rerun to retry them.", parameters=parameters)
