"""Decompose the task into a plan, then drive the plan one step at a time (GameBoyRL
execution/supervisors/subgoal.py).

SubgoalSupervisor is RevisingSupervisor whose targets are plan steps. The extra targets bring the
judge (an intermediate step has no env signal; the last step reverts to the task and is never
judged), the regression check (a step can undo one already cleared) and plan-flaw replanning.
"""
from typing import List, Optional
from cusi_utils.log_handling import log_warn
from cusi_supervisors.parsing import parse_key_value, parse_plan, parse_yes_no
from cusi_supervisors.revising import RevisingSupervisor


class SubgoalSupervisor(RevisingSupervisor):
    """:param max_replans: how many times the plan may be rewritten in one episode."""

    def __init__(self, *, max_replans: int = 2, **kwargs) -> None:
        self.max_replans = max_replans
        self.n_replans = 0
        self.plan: List[str] = []
        self.original_plan: List[str] = []
        self.planned = False
        self.completed_steps: List[dict] = []
        super().__init__(**kwargs)

    def _run_config(self) -> dict:
        return {**super()._run_config(), "max_replans": self.max_replans}

    # -- Planning -------------------------------------------------------------

    def write_plan(self) -> List[str]:
        screen = self.current_frame()
        prompt = (
            self._prompts["PLAN_PROMPT"]
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[INSIGHTS]", self._knowledge() or "(nothing recorded)")
        )
        output = self._vlm_call("plan", texts=prompt, images=[screen])
        self.plan = parse_plan(output)
        return self.plan

    def _resolve_targets(self) -> List[Optional[str]]:
        self.completed_steps = []
        self.n_replans = 0
        self.write_plan()
        self.original_plan = list(self.plan)
        self.planned = bool(self.plan)
        if not self.plan:
            log_warn("[subgoal] no plan produced; running the task unplanned.", parameters=self._parameters)
            return [None]
        return list(self.plan)

    # -- Judging and repair ---------------------------------------------------

    def _on_target_cleared(self, target: str) -> None:
        self.completed_steps.append({"step": target, "frame": self.current_frame()})

    def _check_regression(self, summaries: List[str]) -> Optional[str]:
        """Whether the failed attempt undid the last completed step (its proving frame vs now)."""
        if not self.completed_steps:
            return None
        last = self.completed_steps[-1]
        earlier = self.completed_steps[:-1]
        earlier_block = ""
        if earlier:
            earlier_block = ("Steps completed before that one, which must also still hold:\n"
                             + "\n".join(f"- {s['step']}" for s in earlier) + "\n\n")

        prompt = (
            self._prompts["REGRESSION_CHECK_PROMPT"]
            .replace("[GAME]", self._game)
            .replace("[PREVIOUS_STEP]", last["step"])
            .replace("[EARLIER_STEPS_BLOCK]", earlier_block)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries) or "  (no actions taken)")
        )
        output = self._vlm_call("regression_check", texts=prompt, images=[last["frame"], self.current_frame()])
        if parse_yes_no(output, "Undone") is not True:
            return None
        lost = (parse_key_value(output, "What was lost") or "").strip()
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if lost.upper().startswith("NONE") or not lost:
            lost = reasoning or "progress from an earlier step is no longer visible"
        note = (f"The attempt undid earlier progress: {lost} "
                f"(the step '{last['step']}' was completed and no longer holds).")
        self._say(f"      REGRESSION: {lost}")
        return note

    def _diagnose_failure(self, index: int, targets: List[Optional[str]], history: List[str], judgement: str,
                          regression: Optional[str]) -> Optional[List[str]]:
        """Ask whether the failure is the plan's fault; if so, replan from the current step on."""
        if self.n_replans >= self.max_replans:
            return None
        if all(text is None for text in targets):
            return None

        plan_lines = []
        for i, text in enumerate(targets):
            if text is None:
                continue
            if i < index:
                marker = "  [DONE]"
            elif i == index:
                marker = "  <-- CURRENT, just failed"
            else:
                marker = "  [not yet attempted]"
            plan_lines.append(f"  {i + 1}. {text}{marker}")

        prompt = (
            self._prompts["PLAN_FLAW_PROMPT"]
            .replace("[GAME]", self._game)
            .replace("[OVERALL_TASK]", self._task)
            .replace("[PLAN_BLOCK]", "\n".join(plan_lines))
            .replace("[FAILURE_HISTORY]", "\n\n".join(f"  attempt {i + 1}: {h}" for i, h in enumerate(history)))
            .replace("[JUDGEMENT]", judgement or "(not judged)")
            .replace("[REGRESSION_LINE]", f"\n{regression}\n" if regression else "")
            .replace("[INSIGHTS]", self._knowledge() or "(nothing recorded)")
        )
        output = self._vlm_call("plan_flaw", texts=prompt, images=[self.current_frame()])
        if parse_yes_no(output, "Flawed") is not True:
            return None

        replacement = parse_plan(output)
        if not replacement:
            log_warn("[subgoal] plan judged flawed but no replacement was produced; continuing with the current step.",
                     parameters=self._parameters)
            return None

        self.n_replans += 1
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        self._say(f"      REPLAN ({self.n_replans}/{self.max_replans}): {reasoning}")
        for text in replacement:
            self._say(f"        - {text}")
        return replacement

    def _on_targets_replaced(self, targets: List[Optional[str]]) -> None:
        self.plan = [t for t in targets if t is not None]

    def _extras(self) -> dict:
        return {**super()._extras(), "plan": list(self.plan), "original_plan": list(self.original_plan),
                "planned": self.planned, "n_replans": self.n_replans}
