"""Judging a finished leg, deriving a critique hint, and distilling guidance from frames.

Ported from GameBoyRL: AttemptCheckerSupervisor.process_executor_return + _describe_trajectory
+ _frame_to_call_cutoff (execution/supervisors/checker.py), derive_critique_hint (same file),
and infer_guidance_for_trajectory (vlm_scripts/infer_guidance.py). All work from frames (the
raw screenshot where the environment has one), plus the agent's final answer when it gave one.
"""
from typing import Any, Optional
from cusi.utils.parsing import parse_key_value
from cusi.practice.prompts import (CONSOLIDATE_GUIDANCE_PROMPT, CRITIQUE_CONSOLIDATE_PROMPT, CRITIQUE_SLICE_PROMPT,
                                   DESCRIBE_CONSOLIDATE_PROMPT, DESCRIBE_SLICE_PROMPT, JUDGE_ANSWER_NOTE,
                                   JUDGE_BINARY_PROMPT, JUDGE_GOAL_CONDITION_NOTE, SLICE_GUIDANCE_PROMPT, Domain, fill)
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
    """AttemptCheckerSupervisor._describe_trajectory over the post-step frames."""
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
    """_frame_to_call_cutoff: a 1-based frame number -> how many leading calls to keep."""
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
    """The checker's verdict on a leg: {success, safe_success_point (a call-log cutoff),
    description, reasoning, termination_reason, n_env_steps, max_steps, judge_calls}."""
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


def derive_critique_hint(*, report: LegReport, vlm: AgentVLM, domain: Domain, max_new_tokens: int = 2000,
                         previous_hint: str = "", max_frames_per_slice: int = 8) -> str:
    """derive_critique_hint: critique each window of the failed leg, then consolidate."""
    steps = report.env_steps
    if not steps:
        return previous_hint
    total = len(steps)
    log: list = []
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


def parse_guidance(text: str) -> Optional[dict]:
    """infer_guidance._parse_guidance (lowercases, as GameBoyRL does)."""
    text_lower = text.lower()
    stop = text_lower.find("[stop]")
    if stop != -1:
        text_lower = text_lower[:stop]
    summary = parse_key_value(text_lower, "Summary") or ""
    goal_condition = parse_key_value(text_lower, "Goal condition") or ""
    steps, in_steps = [], False
    for line in text_lower.splitlines():
        stripped = line.strip()
        if stripped.startswith("steps:"):
            in_steps = True
            continue
        if in_steps and stripped.startswith("- "):
            steps.append(stripped[2:].strip())
    if not steps:
        return None
    return {"summary": summary, "goal_condition": goal_condition, "steps": steps}


def infer_guidance(*, frames: list, task: str, vlm: AgentVLM, domain: Domain, max_new_tokens: int = 2000,
                   max_obs_at_once: int = 8) -> Optional[dict]:
    """infer_guidance_for_trajectory over a trajectory's frames (initial + after each step).
    Returns {summary, goal_condition, steps} or None."""
    total = len(frames)
    log: list = []
    ranges = [(s, min(s + max_obs_at_once, total)) for s in range(0, total, max_obs_at_once)]
    prompts = [fill(SLICE_GUIDANCE_PROMPT, domain=domain, TASK=task, START_IDX=s, END_IDX=e - 1, TOTAL=total)
               for s, e in ranges]
    outputs = _batched(vlm, prompts=prompts, images=[frames[s:e] for s, e in ranges], max_new_tokens=max_new_tokens,
                       log=log, tag="guidance_slice")
    sections = []
    for (s, e), output in zip(ranges, outputs):
        parsed = parse_guidance(output)
        if parsed is None:
            continue
        sections.append(f"Frames {s}-{e - 1} / {total}:\nSummary: {parsed['summary']}\nSteps:\n"
                        + "\n".join(f"- {x}" for x in parsed["steps"]))
    if not sections:
        return None
    output = _single(vlm, prompt=fill(CONSOLIDATE_GUIDANCE_PROMPT, domain=domain, TASK=task,
                                      SECTION_DESCRIPTIONS="\n\n".join(sections)),
                     images=None, max_new_tokens=max_new_tokens, log=log, tag="guidance_consolidate")
    return parse_guidance(output)


def format_guidance(guidance: dict) -> str:
    """practice_tasks._format_guidance."""
    lines = []
    if guidance.get("summary"):
        lines.append(f"Summary: {guidance['summary']}")
    if guidance.get("steps"):
        lines.append("Steps:")
        lines += [f"  {i}. {step}" for i, step in enumerate(guidance["steps"], 1)]
    return "\n".join(lines)
