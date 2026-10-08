"""
WebVoyager test-set evaluation through WebVoyagerPlayEnv, matching WebVoyager's run.py and auto_eval.py.
One eval per process: fast_waits is a process-global of WebVoyager's run.py.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import click
import numpy as np
from PIL import Image
from cusi.utils.log_handling import log_info, log_warn
from cusi.eval.records import EvalRow, EvalRun
from cusi.utils.run_dir import run_options
from cusi.utils.paths import eval_episode, eval_run, storage_relative, web_source_task_file, web_task_dir
from cusi.eval.supervision import (run_episode, run_pool, supervisor_extra, supervisor_options, supervisor_settings,
                                   supervisor_state)

MAX_ITER = 15            # run.sh --max_iter
TEMPERATURE = 1.0        # run.sh --temperature
MAX_NEW_TOKENS = 1000    # run.py call_model
MAX_TASK_ATTEMPTS = 3    # run.py --max_task_attempts
JUDGE_IMGS = 15          # evaluation/run_eval.sh --max_attached_imgs
BROWSER_CRASHED = "The browser crashed."   # WebVoyagerPlayEnv's test-mode browser-death error
SCREENSHOTS_FILE = "screenshots.json"     # task dir: run.py screenshot name -> frame in the episode's frames.zip


def load_tasks(*, task_file: str, tasks: str) -> list:
    rows = [json.loads(line) for line in open(task_file) if line.strip()]
    if tasks:
        wanted = [t.strip() for t in tasks.split(",") if t.strip()]
        by_id = {r["id"]: r for r in rows}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            raise click.UsageError(f"Tasks not in {task_file}: {missing}")
        rows = [by_id[t] for t in wanted]
    return rows


def write_artefacts(*, task_dir: str, executors: list, system_prompt: str) -> dict:
    """run.py's chat (the final leg's); returns every leg's screenshots by run.py's name, numbered in episode order."""
    final = executors[-1] if executors else None
    messages = final.messages_for_log() if final is not None and final._messages else \
        [{"role": "system", "content": system_prompt}]
    with open(os.path.join(task_dir, "interact_messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
    screenshots, offset = {}, 0
    for executor in executors:
        for it, frame in executor.screenshots:
            screenshots[f"screenshot{offset + it}.png"] = frame
        offset += max((it for it, _ in executor.screenshots), default=0)
    return screenshots


def materialize_screenshots(*, run_dir: str) -> list:
    """Writes each task dir's screenshots (for auto_eval) from its screenshots.json; returns the paths written."""
    from cusi.agents.frames import read_frame
    written = []
    for task_dir in sorted(glob.glob(os.path.join(run_dir, "task*"))):
        mapping_path = os.path.join(task_dir, SCREENSHOTS_FILE)
        if not os.path.exists(mapping_path):
            continue
        with open(mapping_path) as f:
            mapping = json.load(f)
        for name, ref in mapping.items():
            path = os.path.join(task_dir, name)
            read_frame(path=os.path.join(run_dir, ref)).save(path)
            written.append(path)
    return written


def run_task(*, task: dict, task_dir: str, model_name: str, model_backend: str, vllm_base_url, data_file: str,
             settings: dict, run_dir: str, parameters=None, vlm=None) -> dict:
    """One task with run.py's browser-death retries; vlm overrides the served model (tests)."""
    from cusi.utils.parameter_handling import load_parameters
    from cusi.envs.webvoyager import WebVoyagerPlayEnv, load_webvoyager
    from cusi.eval.episode import write_episode
    from cusi.eval.supervision import model
    from cusi.agents.executors.webvoyager import WebVoyagerExecutor
    parameters = load_parameters(parameters)
    vlm = vlm or model(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url)
    wv = load_webvoyager(project_root=parameters["project_root"])
    t0 = time.time()
    for attempt in range(1, MAX_TASK_ATTEMPTS + 1):
        shutil.rmtree(task_dir, ignore_errors=True)
        os.makedirs(task_dir)
        env, supervisor, result, report, crashed, error = None, None, None, None, None, None
        try:
            env = WebVoyagerPlayEnv(task_id=task["id"], mode="test", data_file=data_file, max_steps=MAX_ITER,
                                    fast_waits=False, headless=True, fix_box_color=True, window_width=1024,
                                    window_height=768, parameters=parameters)
            obs, info = env.reset()

            def make_executor(previous_history=None):
                return WebVoyagerExecutor(env=env, vlm=vlm, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                          previous_history=previous_history, parameters=parameters)

            supervisor, result = run_episode(
                env_name="web", settings=settings, task=task["ques"], env=env, obs=obs, info=info,
                make_executor=make_executor, max_steps=MAX_ITER, parameters=parameters,
                run_kwargs={"env_name": "web", "max_consecutive_invalid": None, "stop_on_model_error": True})
            report = result["report"]
            if any(s.error == BROWSER_CRASHED for r in report.executor_reports for s in r.env_steps):
                crashed = "browser died mid-task"
        except wv.WebDriverException as e:
            if wv._is_browser_dead(e):
                crashed = (str(e).splitlines() or [type(e).__name__])[0][:200]
            else:
                error = f"driver error: {str(e)[:200]}"    # run.py: breaks out of the task
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"   # run.py: an observation error breaks out of the task
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        if crashed is None:
            executors = supervisor.executors if supervisor is not None else []
            screenshots = write_artefacts(task_dir=task_dir, executors=executors, system_prompt=wv.SYSTEM_PROMPT)
            final = report.last_report if report is not None else None
            return _finish(task=task, task_dir=task_dir, run_dir=run_dir, report=report, result=result,
                           screenshots=screenshots,
                           error=error or (final.error if final else None), settings=settings, model_name=model_name,
                           seconds=time.time() - t0)
        log_warn(f"[web eval] task{task['id']}: browser died (attempt {attempt}/{MAX_TASK_ATTEMPTS}): {crashed}",
                 parameters=parameters)
        if attempt == MAX_TASK_ATTEMPTS:
            wv._record_task_error(task_dir, task, f"browser died {MAX_TASK_ATTEMPTS}x: {crashed}")
            return {"termination_reason": None, "answer": None, "n_iterations": 0,
                    "error": f"browser died {MAX_TASK_ATTEMPTS}x: {crashed}", "seconds": time.time() - t0}
        time.sleep(5 * attempt)


def _finish(*, task: dict, task_dir: str, run_dir: str, report, result, screenshots: dict, error, settings: dict,
            model_name: str, seconds: float) -> dict:
    """Writes outcome.json and the episode artifacts; screenshots go in the episode's frames, mapped by name."""
    from cusi.eval.episode import write_episode
    final = report.last_report if report is not None else None
    outcome = {"termination_reason": final.termination_reason if final else None,
               "answer": final.answer if final else None, "error": error,
               "n_iterations": report.n_steps if report else 0, "n_invalid": report.n_invalid if report else 0,
               "input_tokens": ((report.executor_input_tokens or 0) + (report.supervisor_input_tokens or 0))
               if report else 0,
               "output_tokens": ((report.executor_output_tokens or 0) + (report.supervisor_output_tokens or 0))
               if report else 0,
               "seconds": seconds,
               "extra": supervisor_extra(report=report, result=result) if report is not None else {}}
    with open(os.path.join(task_dir, "outcome.json"), "w") as f:
        json.dump(outcome, f, indent=1, default=str)
    if report is None:     # no episode to hold them
        for name, frame in screenshots.items():
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(os.path.join(task_dir, name))
        return outcome
    episode_dir = eval_episode(run_dir=run_dir, task_id=task["id"])
    refs = write_episode(directory=episode_dir, report=report, named_images=screenshots, meta={
            "task_id": task["id"], "task": task["ques"], "env": "web", "start_url": task["web"], "model": model_name,
            "executor": "webvoyager", "supervisor": settings["supervisor"],
            "supervisor_model": settings["supervisor_model"] if settings["supervisor"] != "baseline" else None,
            "termination": outcome["termination_reason"], "budget": MAX_ITER, "answer": outcome["answer"],
            "error": error, "seconds": seconds, "state": supervisor_state(result=result),
            "native_dir": os.path.relpath(task_dir, run_dir)})
    rel = os.path.relpath(episode_dir, run_dir)
    with open(os.path.join(task_dir, SCREENSHOTS_FILE), "w") as f:
        json.dump({name: f"{rel}/{ref}" for name, ref in refs.items()}, f, indent=1)
    return outcome


def judge(*, run_dir: str, judge_model_name: str, judge_model_backend: str, judge_vllm_base_url: str,
          parameters: dict) -> dict:
    """Runs WebVoyager's auto_eval.py on run_dir, with the screenshots written out for it, and returns scores.json."""
    cmd = [sys.executable, "-u", os.path.join(parameters["project_root"], "WebVoyager", "evaluation", "auto_eval.py"),
           "--process_dir", run_dir, "--model_name", judge_model_name, "--model_backend", judge_model_backend,
           "--max_attached_imgs", str(JUDGE_IMGS)]
    if judge_vllm_base_url:
        cmd += ["--vllm_base_url", judge_vllm_base_url]
    written = materialize_screenshots(run_dir=run_dir)
    try:
        with open(os.path.join(run_dir, "auto_eval.log"), "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parameters["project_root"])
    finally:
        for path in written:
            os.remove(path)
    with open(os.path.join(run_dir, "scores.json")) as f:
        return json.load(f)


@click.command()
@click.option("--model_name", required=True, help="Model the agent uses (e.g. the vLLM-served name).")
@click.option("--model_backend", required=True, type=click.Choice(["vllm", "openai", "openrouter"]))
@click.option("--vllm_base_url", default=None)
@click.option("--judge_model_name", required=True, help="auto_eval's judge model.")
@click.option("--judge_model_backend", required=True, type=click.Choice(["vllm", "openai", "openrouter"]))
@click.option("--judge_vllm_base_url", default=None)
@click.option("--run_name", required=True)
@click.option("--task_file", default=None,
              help="Default: WebVoyager/data/WebVoyager_data.jsonl (all 643). 'cusi': that file without the sites "
                   "configs/project_vars.yaml excludes (WebVoyager_data.cusi.jsonl, written from the list).")
@click.option("--tasks", default=None, help="Comma-separated task ids (default: every task in the file).")
@click.option("--skip_judge", is_flag=True, help="Only run the agent; score later.")
@click.option("--workers", default=1, show_default=True, help="Tasks run in parallel (one browser each).")
@supervisor_options
@run_options
@click.pass_obj
def command(parameters, model_name, model_backend, vllm_base_url, judge_model_name, judge_model_backend,
            judge_vllm_base_url, run_name, task_file, tasks, skip_judge, workers, supervisor, supervisor_model,
            supervisor_backend, supervisor_vllm_base_url, max_leg_steps, info_docs, overwrite,
            ignore_config_violation):
    """WebVoyager test-set evaluation through WebVoyagerPlayEnv, under a GameBoyRL supervisor arm."""
    settings = supervisor_settings(supervisor=supervisor, supervisor_model=supervisor_model,
                                   supervisor_backend=supervisor_backend,
                                   supervisor_vllm_base_url=supervisor_vllm_base_url, max_leg_steps=max_leg_steps,
                                   info_docs=info_docs, model_name=model_name, model_backend=model_backend,
                                   vllm_base_url=vllm_base_url)
    if task_file == "cusi":
        from cusi.envs.webvoyager import excluded_sites, filtered_task_file
        task_file = filtered_task_file(parameters=parameters)
        log_info(f"[web eval] task file {task_file}: WebVoyager without {excluded_sites(parameters=parameters)}",
                 parameters=parameters)
    task_file = task_file or web_source_task_file(parameters=parameters)
    rows = load_tasks(task_file=task_file, tasks=tasks)
    log_info(f"[web eval] {len(rows)} tasks from {task_file}", parameters=parameters)
    run_dir = eval_run(parameters=parameters, env="web", run=run_name)
    run = EvalRun(directory=run_dir, config={"env": "web", "model_name": model_name, "model_backend": model_backend,
                                             "judge_model_name": judge_model_name,
                                             "task_file": storage_relative(parameters=parameters, path=task_file),
                                             "tasks": [r["id"] for r in rows], "max_iter": MAX_ITER,
                                             "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS,
                                             "workers": workers, **settings},
                  overwrite=overwrite, ignore_config_violation=ignore_config_violation)
    jobs = []
    for task in rows:
        task_dir = web_task_dir(run_dir=run_dir, task_id=task["id"])
        # Resume as run.py does.
        if os.path.exists(os.path.join(task_dir, "interact_messages.json")) or \
                os.path.exists(os.path.join(task_dir, "task_error.json")):
            continue
        jobs.append(dict(task=task, task_dir=task_dir, model_name=model_name, model_backend=model_backend,
                         vllm_base_url=vllm_base_url, data_file=task_file, settings=settings, run_dir=run_dir,
                         parameters=parameters if workers <= 1 else None))
    log_info(f"[web eval] {run_name}: {len(jobs)} tasks to run ({len(rows)} in the file) under {supervisor}, "
             f"{workers} worker(s) -> {run_dir}", parameters=parameters)

    def on_result(job, out):
        task = job["task"]
        if isinstance(out, Exception):
            log_warn(f"[web eval] task{task['id']} failed to run: {type(out).__name__}: {out}", parameters=parameters)
            return
        log_info(f"[web eval] task{task['id']}: {out['termination_reason']}, {out['n_iterations']} iterations, "
                 f"answer {out['answer']!r} ({out['seconds']:.0f}s)", parameters=parameters)

    t0 = time.time()
    run_pool(jobs=jobs, worker=run_task, workers=workers, on_result=on_result)
    log_info(f"[web eval] agent runs: {time.time() - t0:.0f}s for {len(jobs)} tasks", parameters=parameters)
    if skip_judge:
        log_info(f"[web eval] agent runs done -> {run_dir} (judge skipped)", parameters=parameters)
        return
    scores = judge(run_dir=run_dir, judge_model_name=judge_model_name, judge_model_backend=judge_model_backend,
                   judge_vllm_base_url=judge_vllm_base_url, parameters=parameters)
    done = {r["task_id"] for r in run.rows()}
    for task in rows:
        if task["id"] in done:
            continue
        verdict = scores["tasks"].get(task["id"])
        if verdict is None:
            continue
        outcome_path = os.path.join(web_task_dir(run_dir=run_dir, task_id=task["id"]), "outcome.json")
        outcome = json.load(open(outcome_path)) if os.path.exists(outcome_path) else {}
        run.write(EvalRow(env="web", task_id=task["id"], task=task["ques"],
                          success=1.0 if verdict == "success" else 0.0,
                          n_steps=outcome.get("n_iterations", 0), n_invalid=outcome.get("n_invalid", 0),
                          termination_reason=outcome.get("termination_reason"),
                          input_tokens=outcome.get("input_tokens", 0), output_tokens=outcome.get("output_tokens", 0),
                          seconds=outcome.get("seconds", 0.0), answer=outcome.get("answer"),
                          error=outcome.get("error"), extra={"verdict": verdict, **outcome.get("extra", {})}))
        meta_path = os.path.join(eval_episode(run_dir=run_dir, task_id=task["id"]), "meta.json")
        if os.path.exists(meta_path):     # the judge's verdict into the episode artifacts
            with open(meta_path) as f:
                meta = json.load(f)
            meta.update(success=1.0 if verdict == "success" else 0.0, judge_verdict=verdict,
                        judge_model=judge_model_name)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=1)
    summary = run.summary()
    log_info(f"[web eval] {summary} -> {run_dir}", parameters=parameters)
