"""Stage 5: clean the practice data (GameBoyRL clean_practice).

Over the successful episodes only:
  paraphrase  k paraphrases per unique task (the last is reserved for validation).
  filter      ACCEPT/REJECT each call that would become a dataset row (same episode
              selection and safe-success cutoff as the dataset stage), erring towards ACCEPT;
              the after-step frame is appended when the call produced a step.

Output in practice/: paraphrases.json, clean_decisions.csv (group_idx, attempt, call_idx,
accept, reason). Checkpoints make reruns resume.
"""
import os
import pickle
import pandas as pd
from cusi_utils.lm_inference import parse_key_value
from cusi_utils.log_handling import log_info, log_warn
from cusi_practice.executors.base import strip_hint_blocks, strip_hint_messages
from cusi_practice.parsing import parse_list
from cusi_practice.prompts import (AUGMENT_PARAPHRASE_PROMPT, CLEAN_PROMPT, CLEAN_PROMPT_TRANSITION,
                                   CLEAN_SUMMARY_PROMPT, fill)
from cusi_practice.records import SUMMARY_TAG
from cusi_practice.stages.common import PracticePaths, atomic_json, load_json, run_jobs


def select_successful(df: pd.DataFrame) -> pd.DataFrame:
    """Episodes usable for the dataset: the ones the judge called successful."""
    if df.empty or "success" not in df.columns:
        return df.iloc[0:0]
    return df[df["success"] == True]   # noqa: E712 (pandas)


def episode_cutoff(*, safe_success_point, n_calls: int, safety_margin: int) -> int:
    """_episode_cutoff: leading calls to keep (safe_success_point is a call-log index)."""
    if safe_success_point is None or pd.isna(safe_success_point):
        return n_calls
    return min(n_calls, int(safe_success_point) + safety_margin)


def candidate_calls(*, report, cutoff: int, dataset_tags: tuple) -> list:
    """(call_idx, call) pairs that may become dataset rows: dataset-tagged calls before the
    cutoff whose action the environment accepted (or that finished the leg). A summary row is
    kept when the action call it follows was."""
    out, last_action_ok = [], False
    for call_idx, call in enumerate(report.calls[:cutoff]):
        if call.tag not in dataset_tags:
            continue
        if call.tag == SUMMARY_TAG:
            if last_action_ok:
                out.append((call_idx, call))
            continue
        last_action_ok = call.accepted_by_env()
        if last_action_ok:
            out.append((call_idx, call))
    return out


def render_chat(messages: list) -> str:
    """A chat as text for the reviewer, with <image> where screenshots were."""
    lines = []
    for msg in strip_hint_messages(messages):
        content = msg["content"]
        if isinstance(content, str):
            lines.append(f"[{msg['role']}] {content}")
        else:
            lines.append(f"[{msg['role']}] " + " ".join(p["text"] if p["type"] == "text" else "<image>"
                                                        for p in content))
    return "\n".join(lines)


def _filter_call(*, call, task: str, domain, vlm, max_new_tokens: int) -> tuple:
    images = list(call.images)
    agent_prompt = render_chat(call.messages) if call.messages is not None else strip_hint_blocks(call.prompt)
    if call.tag == SUMMARY_TAG:
        template = CLEAN_SUMMARY_PROMPT
    else:
        steps = call.env_steps
        use_transition = bool(steps) and bool(images)
        if use_transition:
            images.append(steps[-1].frame_after)
        template = CLEAN_PROMPT_TRANSITION if use_transition else CLEAN_PROMPT
    prompt = fill(template, domain=domain, TASK=task, AGENT_PROMPT=agent_prompt, AGENT_RESPONSE=call.response or "")
    output = vlm.infer(texts=prompt, images=images or None, max_new_tokens=max_new_tokens)["output"]
    decision = parse_key_value(output, "Decision") or ""
    return "reject" not in decision.lower(), parse_key_value(output, "Reason") or ""


def clean(*, spec, vlm, paths: PracticePaths, k: int, safety_margin: int, max_new_tokens: int, overwrite: bool,
          parameters: dict) -> None:
    practice_dir = paths.practice_dir
    results_csv = os.path.join(practice_dir, "results.csv")
    if not os.path.exists(results_csv):
        log_warn("clean: no practice/results.csv; run the practice stage first.", parameters=parameters)
        return
    try:
        df = pd.read_csv(results_csv)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame([])
    successful = select_successful(df)
    log_info(f"clean: {len(df)} episodes, {len(successful)} successful.", parameters=parameters)

    # ------------------------------------------------------------ paraphrases
    para_path = os.path.join(practice_dir, "paraphrases.json")
    if not os.path.exists(para_path) or overwrite:
        ckpt = para_path.replace(".json", "_checkpoint.json")
        paraphrases = {} if overwrite else load_json(ckpt, {})
        tasks = [t for t in successful["task_string"].dropna().unique() if t not in paraphrases] \
            if len(successful) else []

        def para(task, *, worker: int):
            output = vlm.infer(texts=fill(AUGMENT_PARAPHRASE_PROMPT, domain=spec.domain, CORE_TASK=task, N=k),
                               max_new_tokens=max_new_tokens)["output"]
            return [p for p in parse_list(output) if p and p != task][:k]

        def save_para(task, result):
            if len(result) < k:
                log_warn(f"clean: only {len(result)} paraphrase(s) for {task!r}.", parameters=parameters)
            paraphrases[task] = result
            atomic_json(paraphrases, ckpt)

        run_jobs(jobs=tasks, fn=para, n_workers=8, desc="paraphrase", on_result=save_para, parameters=parameters)
        atomic_json(paraphrases, para_path)
        if os.path.exists(ckpt):
            os.remove(ckpt)

    # ------------------------------------------------------------ filter
    out_path = os.path.join(practice_dir, "clean_decisions.csv")
    if os.path.exists(out_path) and not overwrite:
        log_info(f"clean: decisions already exist ({out_path}).", parameters=parameters)
        return
    ckpt = out_path.replace(".csv", "_checkpoint.json")
    rows = [] if overwrite else load_json(ckpt, [])
    done = {(r["group_idx"], r["attempt"], r["call_idx"]) for r in rows}
    jobs = []
    for _, row in successful.iterrows():
        group, attempt = str(row["group_idx"]), int(row["attempt"])
        pkl = os.path.join(practice_dir, f"{group}_{attempt}.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            report = pickle.load(f)
        cutoff = episode_cutoff(safe_success_point=row.get("safe_success_point"), n_calls=len(report.calls),
                                safety_margin=safety_margin)
        for call_idx, call in candidate_calls(report=report, cutoff=cutoff, dataset_tags=spec.dataset_tags):
            if (group, attempt, call_idx) not in done:
                jobs.append((group, attempt, call_idx, row["task_string"], call))

    def filt(job, *, worker: int):
        return _filter_call(call=job[4], task=job[3], domain=spec.domain, vlm=vlm, max_new_tokens=max_new_tokens)

    def save(job, result):
        rows.append({"group_idx": job[0], "attempt": job[1], "call_idx": job[2], "tag": job[4].tag,
                     "accept": result[0], "reason": result[1]})
        atomic_json(rows, ckpt)

    failures = run_jobs(jobs=jobs, fn=filt, n_workers=16, desc="filter", on_result=save, parameters=parameters)
    if failures:
        log_warn("clean: some filter calls failed; not finalising. Rerun to retry them.", parameters=parameters)
        return
    pd.DataFrame(rows, columns=["group_idx", "attempt", "call_idx", "tag", "accept", "reason"]).to_csv(
        out_path, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    log_info(f"clean: {len(rows)} decisions ({sum(not r['accept'] for r in rows)} reject) -> {out_path}",
             parameters=parameters)
