"""
AndroidWorld's M3A agent inside the shared skeleton, using M3A's own prompts and step handling.
"""
from typing import Optional
import numpy as np
from android_world.agents import agent_utils, m3a, m3a_utils
from android_world.env import json_action
from cusi.envs.android_world import MSG_EXEC_FAILED
from cusi.agents.executors.base import GUIDANCE_END, GUIDANCE_START, SECRET_NOTE, Decision, Executor

# M3A.step's history messages, verbatim.
MSG_BAD_FORMAT = "Output for action selection is not in the correct format, so no action is performed."
MSG_BAD_ACTION = ("Can not parse the output to a valid action. Please make sure to pick the action from the list"
                  " with required parameters (if any) in the correct JSON format!")


def action_prompt(*, goal: str, history: list, ui_elements: str, guidance: Optional[str]) -> str:
    """M3A's action prompt, with the guidance as a marked additional guideline."""
    if not guidance:
        return m3a._action_selection_prompt(goal, history, ui_elements, None)
    text = f"{guidance}\n{SECRET_NOTE}"
    prompt = m3a._action_selection_prompt(goal, history, ui_elements, [text])
    block = f"For The Current Task:\n- {text}\n"
    assert prompt.count(block) == 1
    return prompt.replace(block, GUIDANCE_START + block + GUIDANCE_END)


class M3AExecutor(Executor):
    """M3A on an AndroidPlayEnv built with som=True."""

    name = "m3a"

    # refresh_before_decide re-observes each step, as M3A does; pair it with stabilize_after_action=False.
    def __init__(self, *, refresh_before_decide: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.refresh_before_decide = refresh_before_decide
        self._summaries: list = []
        self._pending: Optional[dict] = None
        self._before: Optional[tuple] = None

    def reset_memory(self) -> None:
        self._summaries = []
        self._pending = None
        self._before = None

    def history(self) -> list:
        """M3A's step summaries; numbering continues in the next leg."""
        return list(self._summaries)

    def restore_history(self, history: list) -> None:
        self._summaries = list(history)

    def _history(self) -> list:
        return ["Step " + str(i + 1) + "- " + s for i, s in enumerate(self._summaries)]

    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        # Error feedback reaches M3A through its history, not `error`.
        if self.refresh_before_decide:
            obs, info = self.env.observe()
        self._before = (obs, info)
        prompt = action_prompt(goal=self.task, history=self._history(), ui_elements=obs["texts"]["ui_elements"],
                               guidance=hint)
        response = self.call(tag="action", prompt=prompt, images=[info["raw_frame"], obs["frame"]])
        reason, action = m3a_utils.parse_reason_action_output(response)
        if (not reason) or (not action):
            self._summaries.append(MSG_BAD_FORMAT)
            return Decision(invalid="bad format", error=MSG_BAD_FORMAT)
        parsed = agent_utils.extract_json(action)
        try:
            json_action.JSONAction(**parsed)        # M3A's validity test
        except Exception:
            self._summaries.append(MSG_BAD_ACTION)
            return Decision(invalid="unparseable action", error=MSG_BAD_ACTION)
        if parsed.get("action_type") == "status":
            return Decision(finish=True, answer=None, action_text=response)
        self._pending = {"action": action, "reason": reason,
                         "answer": parsed.get("text") if parsed.get("action_type") == "answer" else None}
        return Decision(action_text=response)

    def after_step(self, *, decision, obs_before, info_before, obs_after, info_after, step) -> None:
        pending = self._pending
        self._pending = None
        if pending and pending["answer"]:
            self.report.answer = pending["answer"]
        if not step.valid:
            # As M3A: the error becomes the summary, except a failed execution, which is not recorded.
            if step.error != MSG_EXEC_FAILED:
                self._summaries.append(step.error)
            return
        if self._before is not None:
            obs_before = self._before[0]     # the observation the action was chosen on
        # M3A.step effectively sends an unlabelled "before" and a labelled "after".
        before = np.array(obs_before["frame"], copy=True)
        after = np.array(obs_after["frame"], copy=True)
        m3a_utils.add_screenshot_label(after, "after")
        prompt = m3a._summarize_prompt(pending["action"], pending["reason"], self.task,
                                       obs_before["texts"]["ui_elements"], obs_after["texts"]["ui_elements"])
        current = self._current_call
        summary = self.call(tag="summary", prompt=prompt, images=[before, after])
        self._current_call = current   # the step stays owned by the action call
        self._summaries.append(f"Action selected: {pending['action']}. {summary}")
