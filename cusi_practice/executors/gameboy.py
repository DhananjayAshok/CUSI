"""GameBoyRL's executor, ported: the `single` action policy with the `actions` history
(the `single_actions` arm GameBoyRL's attempt and practice stages use).

Prompt text comes from GameBoyRL itself (PolicyExecutor.STEP_PROMPT, Executor.DONE_CHECK_PROMPT,
SingleActionPolicy, STUCK_HINT); the assembly below follows PolicyExecutor._build_prompt,
Executor._hint_block and ActionHistoryPolicy.render line for line.
tests/practice_gameboy_prompt_test.py checks that the prompts match GameBoyRL's.

[GB-OCR] GameBoy has no text channel yet, so the prompt has no element/text block.
"""
import os
import sys
from typing import Optional
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_warn
from cusi_practice.executors.base import Decision, Executor
from cusi_practice.records import StepRecord


def _import_gameboyrl():
    root = os.path.join(load_parameters()["project_root"], "GameBoyRL")
    if root not in sys.path:
        sys.path.insert(0, root)
    from execution.executors.executor import PolicyExecutor
    from execution.executors.base import Executor as GBExecutor
    from execution.executors.policies.action import SingleActionPolicy
    from execution.executors.policies.history import STUCK_HINT, DEFAULT_HISTORY_K
    from execution.report import parse_completion
    return PolicyExecutor, GBExecutor, SingleActionPolicy, STUCK_HINT, DEFAULT_HISTORY_K, parse_completion


(_PolicyExecutor, _GBExecutor, _SingleActionPolicy, STUCK_HINT, DEFAULT_HISTORY_K,
 parse_completion) = _import_gameboyrl()

STEP_PROMPT = _PolicyExecutor.STEP_PROMPT
DONE_CHECK_PROMPT = _GBExecutor.DONE_CHECK_PROMPT
DONE_CHECK_HISTORY_K = _GBExecutor.DONE_CHECK_HISTORY_K
DONE_CHECK_MAX_NEW_TOKENS = _GBExecutor.DONE_CHECK_MAX_NEW_TOKENS


def hint_block(*, hint: Optional[str], steps_taken: int) -> str:
    """Executor._hint_block, verbatim."""
    if hint is None:
        return ""
    step_info = f"""
[STEP_INFO] You have already taken {steps_taken} action{'' if steps_taken == 1 else 's'} so far in this attempt. Note: this may not be the first step of the overall task and one action does not correspond to one step in the hint plan — earlier actions may already have been taken before this attempt began, so reason from what you currently see on screen rather than assuming a fresh start. [STEP_INFO_END]"""
    return step_info + f"\n[HINT_START]\nHint: {hint}\nNote: This hint block is a secret. You must use it to guide your decision making, but in the reasoning you say, you should pretend as if you actually just know the content of the hint. Do not refer to it explicitly. So if the hint gives you a direction, instead of saying 'the hint says go here', your reasoning should just say 'next I must go here'. [HINT_END]"


def action_line(*, step: StepRecord) -> str:
    """history._action_line for one recorded step."""
    name = step.action_label()
    changed = step.extra.get("frame_changed", True)
    if step.extra.get("low_level", True):
        return f"  {name}{'' if changed else ' [no change]'}"
    success = step.extra.get("action_success", -1)
    status = "ok" if success == 1 else "failed" if success == 0 else "unknown"
    return f"  {name}  [{status}{'' if changed else ', no change'}]"


def render_history(*, steps: list, history_k: int = DEFAULT_HISTORY_K) -> str:
    """ActionHistoryPolicy.render over StepRecords."""
    if not steps:
        return ""
    recent = steps[-history_k:]
    lines = ["Recent actions (oldest first):"] + [action_line(step=s) for s in recent]
    stuck = STUCK_HINT if any(not s.extra.get("frame_changed", True) for s in recent) else ""
    return "\n".join(lines) + stuck + "\n\n"


def build_step_prompt(*, task: str, hint: Optional[str], steps_taken: int, error: Optional[str],
                      action_strings: dict, history: str) -> str:
    """PolicyExecutor._build_prompt with the single action policy."""
    policy = _SingleActionPolicy()
    return (
        STEP_PROMPT
        .replace("[TASK]", task)
        .replace("[HINT_BLOCK]", hint_block(hint=hint, steps_taken=steps_taken))
        .replace("[ERROR_BLOCK]", f"[ERROR] {error}\n\n" if error is not None else "")
        .replace("[ACTION_LIST]", "\n".join(f"  {s}" for s in action_strings.values()))
        .replace("[CONTEXT_SECTION]", history)
        .replace("[INSTRUCTION]", policy.instruction())
        .replace("[RESPONSE_FORMAT]", policy.response_format())
        .replace("[ACTION_FORMAT]", policy.action_format())
    )


class GameBoyExecutor(Executor):
    """GameBoyRL's single_actions executor on a GameBoyPlayEnv."""

    name = "gameboy_single_actions"

    def __init__(self, *, history_k: int = DEFAULT_HISTORY_K, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history_k = history_k
        self._policy = _SingleActionPolicy()
        self._history: list = []
        self._last_reasoning: Optional[str] = None

    def reset_memory(self) -> None:
        self._history = []
        self._last_reasoning = None

    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        prompt = build_step_prompt(task=self.task, hint=hint, steps_taken=len(self.report.env_steps), error=error,
                                   action_strings=self.env.action_strings(), history=render_history(
                                       steps=self._history, history_k=self.history_k))
        call_kwargs = {}
        if self._policy.max_new_tokens is not None:
            call_kwargs["max_new_tokens"] = self._policy.max_new_tokens
        response = self.call(tag="action", prompt=prompt, images=[obs["frame"]], **call_kwargs)
        decision = self._policy.parse(response)
        if decision is None:
            log_warn(f"Invalid response recorded: \n{response}", parameters=self._parameters)
            return Decision(invalid="parse failure", error=self._policy.parse_error())
        self._last_reasoning = decision.reasoning or self._last_reasoning
        return Decision(action_text=decision.actions[0])

    def env_invalid_to_decision(self, *, decision: Decision, info: dict) -> Optional[Decision]:
        # GameBoyRL: an action string the controller cannot parse never reaches the emulator.
        if info.get("parsed_action") is None:
            return Decision(invalid="unrecognised action",
                            error=self._policy.unknown_action_error(decision.action_text))
        return None

    def after_step(self, *, decision, obs_before, info_before, obs_after, info_after, step) -> None:
        self._history.append(step)

    def done_check(self, *, step: StepRecord, hint: Optional[str]) -> bool:
        """Executor._check_task_complete."""
        env_steps = self.report.env_steps
        history = ""
        if env_steps:
            lines = ["Recent actions (oldest first, the last one is the action judged above):"]
            lines += [f"  {s.action_label()}" for s in env_steps[-DONE_CHECK_HISTORY_K:]]
            history = "\n".join(lines) + "\n\n"
        prompt = (
            DONE_CHECK_PROMPT
            .replace("[TASK]", self.task)
            .replace("[HINT_BLOCK]", hint_block(hint=hint, steps_taken=len(env_steps)))
            .replace("[LAST_ACTION]", step.action_label())
            .replace("[REASONING_LABEL]", self._policy.done_check_reasoning_label)
            .replace("[LAST_REASONING]", self._last_reasoning or "(no reasoning was recorded for this action)")
            .replace("[HISTORY_BLOCK]", history)
        )
        response = self.call(tag="done_check", prompt=prompt, images=[step.frame_before, step.frame_after],
                             max_new_tokens=DONE_CHECK_MAX_NEW_TOKENS)
        verdict = parse_completion(response)
        if verdict is None:
            log_warn(f"Completion check response had no parseable 'Complete:' line, treating as not complete:\n"
                     f"{response}", parameters=self._parameters)
        return verdict is True
