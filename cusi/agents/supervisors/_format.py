"""Rendering an executor attempt as text for a supervisor prompt (GameBoyRL
execution/supervisors/_format.py) over our LegReport.

The reasoning line is the executor's own: GameBoy "Reasoning:", M3A "Reason:", WebVoyager
"Thought:" (SupervisorDomain.reasoning_key). An Android / Web step the env rejected is shown with
its error; a finishing action that ended the leg without reaching the env is shown as FINISH.
"""
from typing import Optional
from cusi.agents.records import StepRecord
from cusi.utils.parsing import parse_completion, parse_key_value

#: Tags of the calls that ask for an action (GameBoyRL's ACTION_TAGS; M3A and WebVoyager use "action").
ACTION_TAGS = {"action", "score"}
DONE_CHECK_TAG = "done_check"


def step_name(step, domain) -> str:
    """GameBoyRL report.action_name for one of our steps (plus the env's error on Android / Web;
    GameBoyRL shows an unavailable GameBoy action by its name alone)."""
    if isinstance(step, StepRecord):
        name = step.action_label()
        if domain.show_step_failures and not step.valid:
            name += f" (failed: {step.error})" if step.error else " (failed)"
        return name
    return "INVALID"


def action_trace(report, domain) -> str:
    """Each action the executor took, beside the reasoning it gave for taking it."""
    lines, n_actions = [], 0
    for entry in report.calls:
        if entry.tag == DONE_CHECK_TAG:
            verdict = "yes" if parse_completion(entry.response) is True else "no"
            reason = parse_key_value(entry.response, "Reasoning")
            lines.append(f"            ↳ finished? {verdict}" + (f" — {reason}" if reason else ""))
            continue
        if entry.tag not in ACTION_TAGS:
            continue
        reason = parse_key_value(entry.response, domain.reasoning_key)
        for i, step in enumerate(entry.steps or [None]):
            if step is None:
                name = "FINISH (declared the step done)" if entry.decision == "finish" else "(no action)"
            else:
                name = step_name(step, domain)
            n_actions += 1
            if i == 0:
                lines.append(f"         {n_actions}. {name} — {reason or '(no reasoning given)'}")
            else:
                lines.append(f"         {n_actions}. {name} — (same planned sequence)")
    return "\n".join(lines) if lines else "         (no actions taken)"


def attempt_history_line(report, env_steps: list, summaries: list, hint: Optional[str], verdict: str,
                         domain, regression: Optional[str] = None) -> str:
    """One attempt, described richly enough for the plan reviser to diagnose it."""
    return (
        f"{report.termination_reason} after {len(env_steps)} step(s)\n"
        f"       hint given: {hint or '(none — the step text was the only instruction)'}\n"
        f"       {domain.trace_heading}\n"
        f"{action_trace(report, domain)}\n"
        f"       what visibly happened: "
        f"{' '.join(summaries) if summaries else '(nothing summarised)'}\n"
        + (f"       REGRESSION: {regression}\n" if regression else "")
        + f"       verdict: {verdict}"
    )
