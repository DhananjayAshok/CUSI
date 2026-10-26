"""
Judging a finished leg and deriving a critique hint from a failed one.
"""
from typing import Optional
from cusi.utils.parsing import parse_key_value
from cusi.practice.prompts import (CRITIQUE_CONSOLIDATE_PROMPT, CRITIQUE_SLICE_PROMPT, DESCRIBE_CONSOLIDATE_PROMPT,
                                   DESCRIBE_SLICE_PROMPT, JUDGE_ANSWER_NOTE, JUDGE_BINARY_PROMPT,
                                   JUDGE_GOAL_CONDITION_NOTE, Domain, fill)
from cusi.utils.parsing import parse_int, parse_yes_no
from cusi.agents.records import LegReport
from cusi.agents.vlm import AgentVLM

DESCRIBE_SLICE_SIZE = 10
DEFAULT_LOOKBACK = 8


def _batched(vlm: AgentVLM, *, prompts: list, images: list, max_new_tokens: int, log: list, tag: str) -> list:
    out = vlm.infer(texts=prompts, images=images, max_new_tokens=max_new_tokens)["output"]
    for p, o in zip(prompts, out):
        log.append({"tag": tag, "prompt": p, "response": o})
    return out


def _single(vlm: AgentVLM, *, prompt: str, images: Optional[list], max_new_tokens: int, log: list,
            tag: str) -> str:
    out = vlm.infer(texts=prompt, images=images, max_new_tokens=max_new_tokens)["output"]
    log.append({"tag": tag, "prompt": prompt, "response": out})
    return out


def describe_trajectory(*, frames: list, vlm: AgentVLM, domain: Domain, max_new_tokens: int, log: list) -> str:
    """Describe the post-step frames, in slices consolidated into one description."""
    total = len(frames)
    if total <= DESCRIBE_SLICE_SIZE:
        output = _single(vlm, prompt=fill(DESCRIBE_SLICE_PROMPT, domain=domain, START_IDX=1, END_IDX=total, TOTAL=total),
                         images=frames, max_new_tokens=max_new_tokens, log=log, tag="describe_slice")
        return parse_key_value(output, "Description") or output.strip()
    ranges = [(s, min(s + DESCRIBE_SLICE_SIZE, total)) for s in range(0, total, DESCRIBE_SLICE_SIZE)]
    prompts = [fill(DESCRIBE_SLICE_PROMPT, domain=domain, START_IDX=s + 1, END_IDX=e, TOTAL=total) for s, e in ranges]
    outputs = _batched(vlm, prompts=prompts, images=[frames[s:e] for s, e in ranges], max_new_tokens=max_new_tokens,
                       log=log, tag="describe_slice")
    segments = [f"Frames {s + 1}-{e}: {parse_key_value(o, 'Description') or o.strip()}"
                for (s, e), o in zip(ranges, outputs)]
    output = _single(vlm, prompt=fill(DESCRIBE_CONSOLIDATE_PROMPT, domain=domain,
                                      SEGMENT_DESCRIPTIONS="\n".join(segments)),
                     images=None, max_new_tokens=max_new_tokens, log=log, tag="describe_consolidate")
    return parse_key_value(output, "Description") or output.strip()


def frame_to_call_cutoff(*, calls: list, safe_frame: Optional[int]) -> Optional[int]:
    """A 1-based frame number -> how many leading calls to keep."""
    if safe_frame is None:
        return None
    env_frames = 0
    for call_idx, call in enumerate(calls):
        env_frames += len(call.env_steps)
        if env_frames >= safe_frame:
            return call_idx + 1
    return len(calls)


def judge_leg(*, report: LegReport, vlm: AgentVLM, domain: Domain, lookback: int = DEFAULT_LOOKBACK,
              goal_condition: Optional[str] = None, max_new_tokens: int = 2000) -> dict:
    """The judge's verdict on a leg; safe_success_point is a call-log cutoff."""
    steps = report.env_steps
    log: list = []
    meta = {"termination_reason": report.termination_reason, "n_env_steps": len(steps),
            "max_steps": report.max_steps, "judge_calls": log}
    k = min(lookback, len(steps))
    if k == 0:
        return {"success": False, "safe_success_point": None, "description": "",
                "reasoning": "No environment steps were taken.", **meta}
    frames = [s.frame_after for s in steps]
    description = describe_trajectory(frames=frames, vlm=vlm, domain=domain, max_new_tokens=max_new_tokens, log=log)
    goal_note = fill(JUDGE_GOAL_CONDITION_NOTE, domain=domain, GOAL_CONDITION=goal_condition) if goal_condition else ""
    answer_note = fill(JUDGE_ANSWER_NOTE, domain=domain, ANSWER=report.answer) if report.answer else ""
    prompt = fill(JUDGE_BINARY_PROMPT, domain=domain, TASK=report.task, DESCRIPTION=description,
                  GOAL_CONDITION_NOTE=goal_note, ANSWER_NOTE=answer_note)
    output = _single(vlm, prompt=prompt, images=frames[-k:], max_new_tokens=max_new_tokens, log=log, tag="judge")
    safe_frame = parse_int(output, "Safe success point")
    return {"success": parse_yes_no(output, "Success") is True,
            "safe_success_point": frame_to_call_cutoff(calls=report.calls, safe_frame=safe_frame),
            "description": description, "reasoning": parse_key_value(output, "Reasoning") or "", **meta}


def derive_critique_hint(*, report: LegReport, vlm: AgentVLM, domain: Domain, log: list, max_new_tokens: int = 2000,
                         previous_hint: str = "", max_frames_per_slice: int = 8) -> str:
    """Critique each window of the failed leg, then consolidate into one hint; the calls are appended to log."""
    steps = report.env_steps
    if not steps:
        return previous_hint
    total = len(steps)
    actions = [f"  {i + 1}. {s.action_label()}" for i, s in enumerate(steps)]
    ranges = [(s, min(s + max_frames_per_slice, total)) for s in range(0, total, max_frames_per_slice)]
    prompts = [fill(CRITIQUE_SLICE_PROMPT, domain=domain, TASK=report.task, START_IDX=s + 1, END_IDX=e, TOTAL=total,
                    ACTION_SEQUENCE="\n".join(actions[s:e]) or "  (no actions taken)") for s, e in ranges]
    outputs = _batched(vlm, prompts=prompts, images=[[x.frame_after for x in steps[s:e]] for s, e in ranges],
                       max_new_tokens=max_new_tokens, log=log, tag="critique_slice")
    summaries = []
    for (s, e), output in zip(ranges, outputs):
        for line in output.splitlines():
            if line.strip().lower().startswith("segment summary:"):
                summaries.append(f"Steps {s + 1}-{e}: {line.strip()[len('segment summary:'):].strip()}")
                break
        else:
            summaries.append(f"Steps {s + 1}-{e}: {output.strip()}")
    prior = f'Previous hint (refine or build on this):\n"{previous_hint}"\n\n' if previous_hint else ""
    output = _single(vlm, prompt=fill(CRITIQUE_CONSOLIDATE_PROMPT, domain=domain, TASK=report.task,
                                      SEGMENT_SUMMARIES="\n".join(summaries), PRIOR_HINT_BLOCK=prior),
                     images=None, max_new_tokens=max_new_tokens, log=log, tag="critique_consolidate")
    return parse_key_value(output, "hint") or output.strip()
