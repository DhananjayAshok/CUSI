"""WebVoyager evaluation through WebVoyagerPlayEnv in test mode (eval_plan.md §3.1, §3.4).

    python run_eval.py web --model_name <served name> --model_backend vllm --run_name dev \
        --judge_model_name <served name> --judge_model_backend vllm [--tasks ArXiv--0,GitHub--3 | --task_file f]

Per task, as WebVoyager/run.py with run.sh's flags (--max_iter 15 --max_attached_imgs 3
--temperature 1 --fix_box_color --headless, original waits): a fresh browser on the task's
start page (WebVoyagerPlayEnv, mode="test", fast_waits=False), driven by WebVoyagerExecutor
(run.py's multi-turn chat, 3 screenshots kept, format check, no limit on consecutive invalid
replies), max_new_tokens 1000, temperature 1.0. ANSWER goes through env.step (test mode ends
there). Endings as run.py:
    - a model-call error ends the task with the chat so far (no answer -> judged 0);
    - an observation/driver error that is not a dead browser ends the task the same way;
    - a dead browser reruns the task on a fresh browser, up to 3 times, then task_error.json.
Artefacts as run.py writes them, in <run_dir>/task<id>/: interact_messages.json (the clipped
chat, images as run.py's placeholder) on every ending, and screenshot<it>.png (the set-of-mark
screenshots, numbered by iteration). WebVoyager/evaluation/auto_eval.py then scores <run_dir>
unchanged (--max_attached_imgs 15, the judge model given here), and each task's success comes
from its scores.json (browser_error counts as 0; unjudged counts as 0, as auto_eval does).

Outputs: storage_dir/eval/web/<run_name>/ (results.jsonl, summary.json, scores.json, task dirs).
Needs the CUSI container (Chromium): run via scripts/container.sh. One eval per process:
fast_waits is a process-global of WebVoyager's run.py.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import click
import numpy as np
from PIL import Image
from cusi_utils.log_handling import log_info, log_warn
from cusi_eval.records import EvalRow, EvalRun, call_tokens

MAX_ITER = 15            # run.sh --max_iter
TEMPERATURE = 1.0        # run.sh --temperature
MAX_NEW_TOKENS = 1000    # run.py call_model
MAX_TASK_ATTEMPTS = 3    # run.py --max_task_attempts
JUDGE_IMGS = 15          # evaluation/run_eval.sh --max_attached_imgs
BROWSER_CRASHED = "The browser crashed."   # WebVoyagerPlayEnv's test-mode browser-death error


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


def write_artefacts(*, task_dir: str, executor, system_prompt: str) -> None:
    """interact_messages.json + screenshot<it>.png, as run.py's print_message / save_screenshot."""
    messages = executor.messages_for_log() if executor is not None and executor._messages else \
        [{"role": "system", "content": system_prompt}]
    with open(os.path.join(task_dir, "interact_messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
    for it, frame in (executor.screenshots if executor is not None else []):
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(os.path.join(task_dir, f"screenshot{it}.png"))


def run_task(*, task: dict, task_dir: str, vlm, data_file: str, parameters: dict) -> dict:
    """One task, with run.py's browser-death retries. Returns the outcome fields of the EvalRow."""
    from cusi_envs.webvoyager import WebVoyagerPlayEnv, load_webvoyager
    from cusi_practice.executors.webvoyager import WebVoyagerExecutor
    wv = load_webvoyager(project_root=parameters["project_root"])
    for attempt in range(1, MAX_TASK_ATTEMPTS + 1):
        shutil.rmtree(task_dir, ignore_errors=True)
        os.makedirs(task_dir)
        env, executor, report, crashed, error = None, None, None, None, None
        try:
            env = WebVoyagerPlayEnv(task_id=task["id"], mode="test", data_file=data_file, max_steps=MAX_ITER,
                                    fast_waits=False, headless=True, fix_box_color=True, window_width=1024,
                                    window_height=768, parameters=parameters)
            obs, info = env.reset()
            executor = WebVoyagerExecutor(env=env, vlm=vlm, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                          parameters=parameters)
            report = executor.run(task=task["ques"], hint=None, max_steps=MAX_ITER, obs=obs, info=info, env_name="web",
                                  max_consecutive_invalid=None, finish_through_env=True, stop_on_model_error=True)
            if report.env_steps and report.env_steps[-1].error == BROWSER_CRASHED:
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
            write_artefacts(task_dir=task_dir, executor=executor, system_prompt=wv.SYSTEM_PROMPT)
            return {"report": report, "error": error or (report.error if report else None), "crashed": False}
        log_warn(f"[web eval] task{task['id']}: browser died (attempt {attempt}/{MAX_TASK_ATTEMPTS}): {crashed}",
                 parameters=parameters)
        if attempt == MAX_TASK_ATTEMPTS:
            wv._record_task_error(task_dir, task, f"browser died {MAX_TASK_ATTEMPTS}x: {crashed}")
            return {"report": report, "error": f"browser died {MAX_TASK_ATTEMPTS}x: {crashed}", "crashed": True}
        time.sleep(5 * attempt)


def judge(*, run_dir: str, judge_model_name: str, judge_model_backend: str, judge_vllm_base_url: str,
          parameters: dict) -> dict:
    """WebVoyager/evaluation/auto_eval.py on run_dir, unchanged. Returns its scores.json."""
    cmd = [sys.executable, "-u", os.path.join(parameters["project_root"], "WebVoyager", "evaluation", "auto_eval.py"),
           "--process_dir", run_dir, "--model_name", judge_model_name, "--model_backend", judge_model_backend,
           "--max_attached_imgs", str(JUDGE_IMGS)]
    if judge_vllm_base_url:
        cmd += ["--vllm_base_url", judge_vllm_base_url]
    with open(os.path.join(run_dir, "auto_eval.log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=parameters["project_root"])
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
@click.option("--task_file", default=None, help="Default: WebVoyager/data/WebVoyager_data.jsonl.")
@click.option("--tasks", default=None, help="Comma-separated task ids (default: every task in the file).")
@click.option("--skip_judge", is_flag=True, help="Only run the agent; score later.")
@click.pass_obj
def command(parameters, model_name, model_backend, vllm_base_url, judge_model_name, judge_model_backend,
            judge_vllm_base_url, run_name, task_file, tasks, skip_judge):
    """WebVoyager test-set evaluation through WebVoyagerPlayEnv."""
    from cusi_practice.vlm import PracticeVLM
    task_file = task_file or os.path.join(parameters["project_root"], "WebVoyager", "data", "WebVoyager_data.jsonl")
    rows = load_tasks(task_file=task_file, tasks=tasks)
    run_dir = os.path.join(parameters["storage_dir"], "eval", "web", run_name)
    run = EvalRun(directory=run_dir, config={"env": "web", "model_name": model_name, "model_backend": model_backend,
                                             "judge_model_name": judge_model_name, "task_file": task_file,
                                             "tasks": [r["id"] for r in rows], "max_iter": MAX_ITER,
                                             "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS})
    vlm = PracticeVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                      parameters=parameters)
    pending = {}
    for task in rows:
        task_dir = os.path.join(run_dir, f"task{task['id']}")
        # Resume as run.py: a task with interact_messages.json (or a recorded error) is done.
        if os.path.exists(os.path.join(task_dir, "interact_messages.json")) or \
                os.path.exists(os.path.join(task_dir, "task_error.json")):
            continue
        t0 = time.time()
        out = run_task(task=task, task_dir=task_dir, vlm=vlm, data_file=task_file, parameters=parameters)
        report = out["report"]
        n_in, n_out = call_tokens(report) if report is not None else (0, 0)
        pending[task["id"]] = {"task": task, "report": report, "error": out["error"], "crashed": out["crashed"],
                               "tokens": (n_in, n_out), "seconds": time.time() - t0}
        log_info(f"[web eval] task{task['id']}: {report.termination_reason if report else 'no report'}, "
                 f"{len(report.steps) if report else 0} iterations, answer {report.answer if report else None!r} "
                 f"({time.time() - t0:.0f}s)", parameters=parameters)
        with open(os.path.join(task_dir, "outcome.json"), "w") as f:
            json.dump({"termination_reason": report.termination_reason if report else None,
                       "answer": report.answer if report else None, "error": out["error"],
                       "n_iterations": len(report.steps) if report else 0,
                       "n_invalid": sum(1 for s in report.steps if not getattr(s, "valid", False)) if report else 0,
                       "input_tokens": n_in, "output_tokens": n_out, "seconds": time.time() - t0}, f, indent=1)
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
        outcome_path = os.path.join(run_dir, f"task{task['id']}", "outcome.json")
        outcome = json.load(open(outcome_path)) if os.path.exists(outcome_path) else {}
        run.write(EvalRow(env="web", task_id=task["id"], task=task["ques"],
                          success=1.0 if verdict == "success" else 0.0,
                          n_steps=outcome.get("n_iterations", 0), n_invalid=outcome.get("n_invalid", 0),
                          termination_reason=outcome.get("termination_reason"),
                          input_tokens=outcome.get("input_tokens", 0), output_tokens=outcome.get("output_tokens", 0),
                          seconds=outcome.get("seconds", 0.0), answer=outcome.get("answer"),
                          error=outcome.get("error"), extra={"verdict": verdict}))
    summary = run.summary()
    log_info(f"[web eval] {summary} -> {run_dir}", parameters=parameters)
