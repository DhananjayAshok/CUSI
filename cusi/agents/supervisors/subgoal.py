"""
The GameBoy supervisor (decisions.md E7, E8.0): GameBoyRL's subgoal supervisor. It plans the task into steps that
each end in a visible condition, then drives the plan one step at a time.

- A step leg runs the step as the executor's task; the executor's done-check may end it early.
- After each leg, a judge reads windowed summaries of the leg's frames plus the current screen, and decides whether
  the step is done.
- On a failure: check whether earlier progress was undone, ask whether the plan itself is flawed (and replan if so),
  otherwise write a resume hint from the current screen and try again.
- The last step runs the real task, with the step as its hint; only the env can clear it.

task None (exploring): no plan, no judge and no hints; the executor plays one leg with the whole budget.
"""
from typing import List, Optional
from cusi.utils.log_handling import log_warn
from cusi.utils.parsing import parse_completion, parse_key_value, parse_plan, parse_yes_no
from cusi.agents.records import StepRecord
from cusi.agents.supervisors.base import EPISODE_ENDINGS, Supervisor
from cusi.agents.supervisors.prompts import (JUDGE_CONSOLIDATE_PROMPT, JUDGE_SLICE_PROMPT, PLAN_FLAW_PROMPT,
                                             PLAN_PROMPT, REGRESSION_CHECK_PROMPT, RESUME_HINT_PROMPT)

LEG_STEPS = 10                 # GameBoyRL uses 5; 10 is what E7 ran
ATTEMPTS_PER_STEP = 3
FINAL_ATTEMPT_MULTIPLIER = 3   # the last step gets max(ATTEMPTS_PER_STEP * this, budget) attempts
MAX_REPLANS = 2                # per episode
FRAMES_PER_SLICE = 8
HISTORY_ATTEMPTS = 3           # failed attempts the plan review sees
NO_INSIGHTS = "(nothing recorded)"
ACTION_TAG, DONE_CHECK_TAG = "action", "done_check"
TRACE_HEADING = "what the player pressed, and why they said they pressed it:"
TRACE_BLOCK_HEADING = "What they pressed, and the reason they gave for each:"
REGRESSION_TAIL = "the player cannot finish this step from a state they have gone backwards into."


def budget_spent(report) -> int:
    """Recorded steps plus any finish that never reached the env; at least 1."""
    unrecorded_finishes = sum(1 for c in report.calls if c.decision == "finish" and not c.steps)
    return max(1, len(report.steps) + unrecorded_finishes)


def step_name(step) -> str:
    return step.action_label() if isinstance(step, StepRecord) else "INVALID"


def action_trace(report) -> str:
    """Each action the executor took, beside the reasoning it gave for taking it, and its done-check verdicts."""
    lines, n_actions = [], 0
    for entry in report.calls:
        if entry.tag == DONE_CHECK_TAG:
            verdict = "yes" if parse_completion(entry.response) is True else "no"
            reason = parse_key_value(entry.response, "Reasoning")
            lines.append(f"            ↳ finished? {verdict}" + (f" — {reason}" if reason else ""))
            continue
        if entry.tag != ACTION_TAG:
            continue
        reason = parse_key_value(entry.response, "Reasoning")
        for i, step in enumerate(entry.steps or [None]):
            if step is None:
                name = "FINISH (declared the step done)" if entry.decision == "finish" else "(no action)"
            else:
                name = step_name(step)
            n_actions += 1
            if i == 0:
                lines.append(f"         {n_actions}. {name} — {reason or '(no reasoning given)'}")
            else:
                lines.append(f"         {n_actions}. {name} — (same planned sequence)")
    return "\n".join(lines) if lines else "         (no actions taken)"


def attempt_history_line(*, report, env_steps: list, summaries: list, hint: Optional[str], verdict: str,
                         regression: Optional[str]) -> str:
    """One failed attempt, described richly enough for the plan review to diagnose it."""
    return (
        f"{report.termination_reason} after {len(env_steps)} step(s)\n"
        f"       hint given: {hint or '(none — the step text was the only instruction)'}\n"
        f"       {TRACE_HEADING}\n"
        f"{action_trace(report)}\n"
        f"       what visibly happened: "
        f"{' '.join(summaries) if summaries else '(nothing summarised)'}\n"
        + (f"       REGRESSION: {regression}\n" if regression else "")
        + f"       verdict: {verdict}"
    )


def plan_block(*, targets: list, index: int) -> str:
    """The plan with progress marked, for the plan review."""
    lines = []
    for i, text in enumerate(targets):
        if text is None:
            continue
        marker = "  [DONE]" if i < index else "  <-- CURRENT, just failed" if i == index else "  [not yet attempted]"
        lines.append(f"  {i + 1}. {text}{marker}")
    return "\n".join(lines)


class SubgoalSupervisor(Supervisor):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._game = getattr(self._env, "game", "")
        self.plan: List[str] = []
        self.original_plan: List[str] = []
        self.planned = False
        self.n_replans = 0
        self.completed_steps: List[dict] = []
        self.step_log: List[dict] = []
        self.last_diagnosis: Optional[str] = None

    @property
    def max_final_attempts(self) -> int:
        return max(ATTEMPTS_PER_STEP * FINAL_ATTEMPT_MULTIPLIER, self._max_steps)

    def _run_config(self) -> dict:
        return {**super()._run_config(), "leg_steps": LEG_STEPS, "attempts_per_step": ATTEMPTS_PER_STEP,
                "max_final_attempts": self.max_final_attempts, "max_replans": MAX_REPLANS,
                "frames_per_slice": FRAMES_PER_SLICE, "history_attempts": HISTORY_ATTEMPTS}

    def current_frame(self):
        return self._env._env.get_info()["core"]["current_frame"]

    # ------------------------------------------------------------ plan

    def write_plan(self) -> List[str]:
        prompt = (PLAN_PROMPT.replace("[GAME]", self._game).replace("[TASK]", self._task)
                  .replace("[INSIGHTS]", NO_INSIGHTS))
        self.plan = parse_plan(self._vlm_call("plan", texts=prompt, images=[self.current_frame()]))
        return self.plan

    def _resolve_targets(self) -> List[Optional[str]]:
        """The plan's steps, or [None] (the task unplanned) if no plan came back."""
        self.write_plan()
        self.original_plan = list(self.plan)
        self.planned = bool(self.plan)
        if not self.plan:
            log_warn("[subgoal] no plan produced; running the task unplanned.", parameters=self._parameters)
            return [None]
        return list(self.plan)

    def _leg_spec(self, *, target: Optional[str], is_last: bool, hint: Optional[str]) -> tuple:
        """(task, hint, self_terminate); the last target reverts to the real task, hinted with the step."""
        if target is None:
            return self._task, hint, False
        if is_last:
            return self._task, hint or target, False
        return target, hint, True

    # ------------------------------------------------------------ judging

    def _segment_summaries(self, *, env_steps: list, target: str) -> List[str]:
        """One "Steps a-b: ..." summary per window of FRAMES_PER_SLICE frames, in one batched call."""
        if not env_steps:
            return []
        total, ranges, prompts, images = len(env_steps), [], [], []
        names = [f"  {i + 1}. {step_name(s)}" for i, s in enumerate(env_steps)]
        for start in range(0, total, FRAMES_PER_SLICE):
            end = min(start + FRAMES_PER_SLICE, total)
            prompts.append(JUDGE_SLICE_PROMPT.replace("[GAME]", self._game).replace("[TASK]", target)
                           .replace("[START_IDX]", str(start + 1)).replace("[END_IDX]", str(end))
                           .replace("[TOTAL]", str(total))
                           .replace("[ACTION_SEQUENCE]", "\n".join(names[start:end]) or "  (no actions taken)"))
            images.append([s.frame_after for s in env_steps[start:end]])
            ranges.append((start, end))
        outputs = self._vlm_call("judge_slice", texts=prompts, images=images)
        summaries = []
        for (start, end), output in zip(ranges, outputs):
            for line in output.splitlines():
                if line.strip().lower().startswith("segment summary:"):
                    summaries.append(f"Steps {start + 1}-{end}: {line.strip()[len('segment summary:'):].strip()}")
                    break
            else:
                summaries.append(f"Steps {start + 1}-{end}: {output.strip()}")
        return summaries

    def judge_step(self, *, env_steps: list, target: str, stop_reason: str) -> tuple:
        """(complete, reasoning, segment summaries) for whether target was reached."""
        if not env_steps:
            return False, "No environment steps were taken in this attempt.", []
        summaries = self._segment_summaries(env_steps=env_steps, target=target)
        prompt = (JUDGE_CONSOLIDATE_PROMPT.replace("[GAME]", self._game).replace("[TASK]", target)
                  .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries))
                  .replace("[STOP_REASON]", stop_reason or "unknown"))
        output = self._vlm_call("judge", texts=prompt, images=[self.current_frame()])
        verdict = parse_completion(output)
        if verdict is None:
            log_warn("[subgoal] the step judge gave no 'Complete:' line; treating the step as not complete.",
                     parameters=self._parameters)
        return verdict is True, (parse_key_value(output, "Reasoning") or "").strip(), summaries

    def _check_regression(self, *, summaries: List[str]) -> Optional[str]:
        """A note if the failed attempt undid the last completed step, else None."""
        if not self.completed_steps:
            return None
        last, earlier = self.completed_steps[-1], self.completed_steps[:-1]
        earlier_block = ""
        if earlier:
            earlier_block = ("Steps completed before that one, which must also still hold:\n"
                             + "\n".join(f"- {s['step']}" for s in earlier) + "\n\n")
        prompt = (REGRESSION_CHECK_PROMPT.replace("[GAME]", self._game).replace("[PREVIOUS_STEP]", last["step"])
                  .replace("[EARLIER_STEPS_BLOCK]", earlier_block)
                  .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries) or "  (no actions taken)"))
        output = self._vlm_call("regression_check", texts=prompt, images=[last["frame"], self.current_frame()])
        if parse_yes_no(output, "Undone") is not True:
            return None
        lost = (parse_key_value(output, "What was lost") or "").strip()
        if lost.upper().startswith("NONE") or not lost:
            lost = ((parse_key_value(output, "Reasoning") or "").strip()
                    or "progress from an earlier step is no longer visible")
        return (f"The attempt undid earlier progress: {lost} "
                f"(the step '{last['step']}' was completed and no longer holds).")

    def _review_plan(self, *, index: int, targets: list, history: List[str], judgement: str,
                     regression: Optional[str]) -> Optional[List[str]]:
        """A replacement for the current step onwards if the plan itself is judged flawed, else None."""
        if self.n_replans >= MAX_REPLANS or all(text is None for text in targets):
            return None
        prompt = (PLAN_FLAW_PROMPT.replace("[GAME]", self._game).replace("[OVERALL_TASK]", self._task)
                  .replace("[PLAN_BLOCK]", plan_block(targets=targets, index=index))
                  .replace("[FAILURE_HISTORY]", "\n\n".join(f"  attempt {i + 1}: {h}" for i, h in enumerate(history)))
                  .replace("[JUDGEMENT]", judgement or "(not judged)")
                  .replace("[REGRESSION_LINE]", f"\n{regression}\n" if regression else "")
                  .replace("[INSIGHTS]", NO_INSIGHTS))
        output = self._vlm_call("plan_flaw", texts=prompt, images=[self.current_frame()])
        if parse_yes_no(output, "Flawed") is not True:
            return None
        replacement = parse_plan(output)
        if not replacement:
            log_warn("[subgoal] plan judged flawed but no replacement was produced; continuing with the current step.",
                     parameters=self._parameters)
            return None
        self.n_replans += 1
        return replacement

    def write_resume_hint(self, *, summaries: List[str], target: str, judgement: str, previous_hint: str, report,
                          regression: Optional[str]) -> Optional[str]:
        """The hint for the next attempt, anchored on the current screen."""
        prior_block = (f'Previous hint, which did not work (do not simply repeat it):\n"{previous_hint}"\n\n'
                       if previous_hint else "")
        trace_block = f"\n\n{TRACE_BLOCK_HEADING}\n{action_trace(report)}"
        prompt = (RESUME_HINT_PROMPT.replace("[GAME]", self._game).replace("[TASK]", target)
                  .replace("[SEGMENT_SUMMARIES]", ("\n".join(summaries) or "  (no actions taken)") + trace_block)
                  .replace("[JUDGEMENT]", judgement or "the screen does not show the step complete")
                  .replace("[REGRESSION_BLOCK]",
                           f"**{regression} Say what to do about that first — {REGRESSION_TAIL}**\n\n"
                           if regression else "")
                  .replace("[INSIGHTS]", NO_INSIGHTS)
                  .replace("[PRIOR_HINT_BLOCK]", prior_block))
        output = self._vlm_call("hint", texts=prompt, images=[self.current_frame()])
        self.last_diagnosis = (parse_key_value(output, "Diagnosis") or "").strip() or None
        return (parse_key_value(output, "Hint") or "").strip() or None

    # ------------------------------------------------------------ the loop

    def _after_failure(self, *, index: int, targets: list, history: List[str], attempt: dict, report,
                       env_steps: list, summaries: List[str], verdict: str, hint_target: str, hint_reason: str,
                       hint: Optional[str], attempts_at_target: int, attempt_cap: int) -> tuple:
        """(replacement targets or None, next hint, exhausted) after a failed attempt."""
        regression = self._check_regression(summaries=summaries)
        if regression:
            attempt["regression"] = regression
        history.append(attempt_history_line(report=report, env_steps=env_steps, summaries=summaries,
                                            hint=attempt["hint"], verdict=verdict, regression=regression))
        del history[:-HISTORY_ATTEMPTS]
        replacement = self._review_plan(index=index, targets=targets, history=history, judgement=verdict,
                                        regression=regression)
        if replacement:
            return replacement, hint, False
        if attempts_at_target >= attempt_cap:
            return None, hint, True
        hint = self.write_resume_hint(summaries=summaries, target=hint_target, judgement=hint_reason,
                                      previous_hint=hint or "", report=report, regression=regression) or hint
        if self.last_diagnosis:
            attempt["diagnosis"] = self.last_diagnosis
        return None, hint, False

    def _evaluate(self) -> dict:
        if self._task is None:
            self.call_executor(task=None, hint=None, max_steps=self._max_steps)
            return self._extras()
        budget = self._max_steps
        targets = self._resolve_targets()
        index = 0
        while index < len(targets):
            if budget <= 0:
                break
            target = targets[index]
            is_last = index == len(targets) - 1
            record = {"step": target, "attempts": [], "replans": [], "cleared": False}
            hint: Optional[str] = None
            history: List[str] = []
            attempts_at_target = 0

            while budget > 0:
                leg_task, leg_hint, self_terminate = self._leg_spec(target=target, is_last=is_last, hint=hint)
                attempts_at_target += 1
                report = self.call_executor(task=leg_task, hint=leg_hint, max_steps=min(LEG_STEPS, budget),
                                            target=target, final=is_last, self_terminate=self_terminate)
                env_steps = report.env_steps
                budget -= budget_spent(report)
                attempt = {"termination_reason": report.termination_reason, "n_steps": len(env_steps),
                           "hint": leg_hint}

                # The env ending outranks every judgement (max_invalid is re-hintable).
                if report.termination_reason in EPISODE_ENDINGS:
                    attempt["cleared"] = record["cleared"] = report.termination_reason == "terminated"
                    record["attempts"].append(attempt)
                    index = len(targets)          # leave the outer loop too
                    break

                if is_last:
                    attempt["cleared"] = False
                    record["attempts"].append(attempt)
                    summaries = self._segment_summaries(env_steps=env_steps, target=self._task)
                    verdict = "the environment did not signal the task complete"
                    hint_target, hint_reason = self._task, verdict
                    attempt_cap = self.max_final_attempts
                else:
                    complete, reasoning, summaries = self.judge_step(env_steps=env_steps, target=target,
                                                                     stop_reason=report.termination_reason)
                    attempt["cleared"], attempt["judgement"] = complete, reasoning
                    record["attempts"].append(attempt)
                    if complete:
                        record["cleared"] = True
                        self.completed_steps.append({"step": target, "frame": self.current_frame()})
                        break
                    verdict = f"judged incomplete — {reasoning}"
                    hint_target, hint_reason = target, reasoning
                    attempt_cap = ATTEMPTS_PER_STEP

                replacement, hint, exhausted = self._after_failure(
                    index=index, targets=targets, history=history, attempt=attempt, report=report,
                    env_steps=env_steps, summaries=summaries, verdict=verdict, hint_target=hint_target,
                    hint_reason=hint_reason, hint=hint, attempts_at_target=attempts_at_target,
                    attempt_cap=attempt_cap)
                if replacement:
                    record["replans"].append(replacement)
                    targets[index:] = replacement
                    self.plan = [t for t in targets if t is not None]
                    target = targets[index]
                    is_last = index == len(targets) - 1
                    record["step"] = target
                    hint, history, attempts_at_target = None, [], 0
                    continue
                if exhausted:
                    break

            self.step_log.append(record)
            index += 1
        return self._extras()

    def _extras(self) -> dict:
        return {"step_log": self.step_log, "plan": list(self.plan), "original_plan": list(self.original_plan),
                "planned": self.planned, "n_replans": self.n_replans}
