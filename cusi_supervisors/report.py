"""The record of one supervised episode: GameBoyRL's SupervisorReport / SupervisorVLMCallRecord
(execution/report.py) over our LegReport.

    SupervisorCall    one supervisor model call: stage, images, prompt, response, tokens, seconds
    LegEvent          one executor leg: its LegReport plus the target it served, whether it was the
                      final target, whether it could end itself, and its wall-clock seconds
    SupervisorReport  event_log (SupervisorCall and LegEvent interleaved, in order), the supervisor's
                      settings, and the views GameBoyRL has (supervisor_calls, legs, token sums)

cusi_eval.episode writes it to disk as JSON + PNG.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from cusi_practice.records import InvalidRecord, LegReport


def sum_optional(values) -> Optional[int]:
    """GameBoyRL utils.sum_optional: the sum of the known values, None if none is known."""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


@dataclass
class SupervisorCall:
    stage: str
    images: list
    prompt: str
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    seconds: Optional[float] = None


@dataclass
class LegEvent:
    report: LegReport
    #: The plan step this leg attempted; None for the task itself.
    target: Optional[str]
    final: bool
    self_terminate: bool
    seconds: float = 0.0


@dataclass
class SupervisorReport:
    task: str
    supervisor_name: str
    env: str
    init_kwargs: dict = field(default_factory=dict)
    event_log: list = field(default_factory=list)
    narrative: list = field(default_factory=list)
    digest: Optional[str] = None

    @property
    def supervisor_calls(self) -> list:
        return [e for e in self.event_log if isinstance(e, SupervisorCall)]

    @property
    def legs(self) -> list:
        return [e for e in self.event_log if isinstance(e, LegEvent)]

    @property
    def executor_reports(self) -> list:
        return [e.report for e in self.legs]

    @property
    def last_report(self) -> Optional[LegReport]:
        legs = self.executor_reports
        return legs[-1] if legs else None

    @property
    def n_invalid(self) -> int:
        """Decisions that never reached the env, plus env steps the env rejected, over every leg."""
        return sum(1 for r in self.executor_reports for s in r.steps
                   if isinstance(s, InvalidRecord) or not getattr(s, "valid", True))

    @property
    def n_steps(self) -> int:
        """Recorded steps (env steps and invalid decisions) over every leg."""
        return sum(len(r.steps) for r in self.executor_reports)

    @property
    def supervisor_input_tokens(self) -> Optional[int]:
        return sum_optional(c.input_tokens for c in self.supervisor_calls)

    @property
    def supervisor_output_tokens(self) -> Optional[int]:
        return sum_optional(c.output_tokens for c in self.supervisor_calls)

    @property
    def executor_input_tokens(self) -> Optional[int]:
        return sum_optional(c.input_tokens for r in self.executor_reports for c in r.calls)

    @property
    def executor_output_tokens(self) -> Optional[int]:
        return sum_optional(c.output_tokens for r in self.executor_reports for c in r.calls)

    def timing(self) -> dict:
        """Model seconds (supervisor / executor) and leg seconds; env time = leg - executor model."""
        sup = sum(c.seconds or 0.0 for c in self.supervisor_calls)
        exe = sum(c.seconds or 0.0 for r in self.executor_reports for c in r.calls)
        legs = sum(e.seconds for e in self.legs)
        return {"supervisor_model_seconds": sup, "executor_model_seconds": exe, "leg_seconds": legs,
                "env_seconds": max(0.0, legs - exe)}

    def counts(self) -> dict:
        return {"n_legs": len(self.legs), "n_supervisor_calls": len(self.supervisor_calls),
                "n_executor_calls": sum(len(r.calls) for r in self.executor_reports),
                "supervisor_input_tokens": self.supervisor_input_tokens or 0,
                "supervisor_output_tokens": self.supervisor_output_tokens or 0,
                "executor_input_tokens": self.executor_input_tokens or 0,
                "executor_output_tokens": self.executor_output_tokens or 0}

    def __str__(self) -> str:
        """The interleaved event log as text (GameBoyRL's SupervisorReport.__str__ layout)."""
        if not self.event_log:
            return "  (no supervisor events recorded)"
        lines: list = []
        leg = 0
        for event in self.event_log:
            if isinstance(event, SupervisorCall):
                lines.append(f"===== SUPERVISOR [{event.stage}] =====")
                lines.append(_indent(f"Prompt:\n{event.prompt}"))
                lines.append(_indent(f"Response:\n{event.response}"))
            else:
                leg += 1
                lines.append(f"===== EXECUTOR LEG {leg}: {event.report.task!r} =====")
                if event.report.hint:
                    lines.append(_indent(f"hint: {event.report.hint}"))
                lines.append(_indent(f"-> {event.report.termination_reason} after {len(event.report.steps)} step(s)"))
        return "\n".join(lines)


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def jsonable(value: Any) -> Any:
    """step_log / extras values as JSON (images dropped)."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items() if k != "frame"}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
