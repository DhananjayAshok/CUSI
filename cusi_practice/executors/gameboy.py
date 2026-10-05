"""GameBoyRL's executor on a GameBoyPlayEnv: every arm, with GameBoyRL's own policy objects.

An arm is `<action>_<history>` as in GameBoyRL (execution/registry.py):
    action   single | scored | sequence   (execution/executors/policies/action.py)
    history  none | actions | visual      (execution/executors/policies/history.py)
CUSI's default everywhere is single_visual.

The action and history policies are GameBoyRL's own classes, used as is: they build the
instruction / response format, parse replies, and remember steps (the history policy is fed
GameBoyRL EnvironmentStepRecords, and the visual history makes its own batched frame-diff
calls through this executor). The prompt is assembled as PolicyExecutor._build_prompt and the
loop in run() follows PolicyExecutor._execute / _run_decision line for line: a decision may
carry several actions (sequence), an unrecognised action string invalidates and aborts the
decision, a failed high-level action aborts a multi-action decision, terminated/truncated are
checked after every action, the history sees a decision's steps once at its end, and the
completion check (allow_done_check) runs only after a decision that ran to the end.
An action the controller reports unavailable is a normal step, as GameBoyRL's loop never reads
that flag. tests/practice_gameboy_prompt_test.py checks the prompts against GameBoyRL's.

[GB-OCR] GameBoy has no text channel yet, so the prompt has no element/text block.
"""
import os
import sys
from typing import Optional
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_warn
from cusi_practice.executors.base import MAX_CONSECUTIVE_INVALID, Decision, Executor
from cusi_practice.records import CallRecord, EncodedImage, InvalidRecord, LegReport, StepRecord


def _import_gameboyrl():
    root = os.path.join(load_parameters()["project_root"], "GameBoyRL")
    if root not in sys.path:
        sys.path.insert(0, root)
    from execution.executors import executor as gb_executor
    from execution.executors.base import Executor as GBExecutor
    from execution.executors.policies import action as gb_action, history as gb_history
    from execution.report import EnvironmentStepRecord, parse_completion
    return gb_executor, GBExecutor, gb_action, gb_history, EnvironmentStepRecord, parse_completion


_gb_executor, _GBExecutor, _gb_action, _gb_history, GBEnvironmentStepRecord, parse_completion = _import_gameboyrl()

STEP_PROMPT = _gb_executor.PolicyExecutor.STEP_PROMPT
DONE_CHECK_PROMPT = _GBExecutor.DONE_CHECK_PROMPT
DONE_CHECK_HISTORY_K = _GBExecutor.DONE_CHECK_HISTORY_K
DONE_CHECK_MAX_NEW_TOKENS = _GBExecutor.DONE_CHECK_MAX_NEW_TOKENS
DONE_CHECK_EVERY_K_STEPS = _GBExecutor.DONE_CHECK_EVERY_K_STEPS
STUCK_HINT = _gb_history.STUCK_HINT
DEFAULT_HISTORY_K = _gb_history.DEFAULT_HISTORY_K
ACTION_POLICIES = _gb_action.AVAILABLE_ACTION_POLICIES
HISTORY_POLICIES = _gb_history.AVAILABLE_HISTORY_POLICIES
#: Every GameBoyRL arm, "<action>_<history>".
ARMS = tuple(f"{a}_{h}" for a in ACTION_POLICIES for h in HISTORY_POLICIES)
DEFAULT_ARM = "single_visual"


def split_arm(arm: str) -> tuple:
    action, history = arm.split("_", 1)
    if action not in ACTION_POLICIES or history not in HISTORY_POLICIES:
        raise ValueError(f"unknown GameBoyRL executor arm {arm!r}; one of {ARMS}")
    return action, history


def hint_block(*, hint: Optional[str], steps_taken: int) -> str:
    """Executor._hint_block, verbatim."""
    if hint is None:
        return ""
    step_info = f"""
[STEP_INFO] You have already taken {steps_taken} action{'' if steps_taken == 1 else 's'} so far in this attempt. Note: this may not be the first step of the overall task and one action does not correspond to one step in the hint plan — earlier actions may already have been taken before this attempt began, so reason from what you currently see on screen rather than assuming a fresh start. [STEP_INFO_END]"""
    return step_info + f"\n[HINT_START]\nHint: {hint}\nNote: This hint block is a secret. You must use it to guide your decision making, but in the reasoning you say, you should pretend as if you actually just know the content of the hint. Do not refer to it explicitly. So if the hint gives you a direction, instead of saying 'the hint says go here', your reasoning should just say 'next I must go here'. [HINT_END]"


def action_line(*, step: StepRecord) -> str:
    """history._action_line for one of our StepRecords."""
    name = step.action_label()
    changed = step.extra.get("frame_changed", True)
    if step.extra.get("low_level", True):
        return f"  {name}{'' if changed else ' [no change]'}"
    success = step.extra.get("action_success", -1)
    status = "ok" if success == 1 else "failed" if success == 0 else "unknown"
    return f"  {name}  [{status}{'' if changed else ', no change'}]"


def render_history(*, steps: list, history_k: int = DEFAULT_HISTORY_K) -> str:
    """ActionHistoryPolicy.render over our StepRecords (for prompters that keep StepRecords,
    e.g. cusi_explore's). Same slicing as GameBoyRL, including history_k = 0 (all steps)."""
    if not steps:
        return ""
    recent = steps[-history_k:]
    lines = ["Recent actions (oldest first):"] + [action_line(step=s) for s in recent]
    stuck = STUCK_HINT if any(not s.extra.get("frame_changed", True) for s in recent) else ""
    return "\n".join(lines) + stuck + "\n\n"


def build_step_prompt(*, task: str, hint: Optional[str], steps_taken: int, error: Optional[str],
                      action_strings: dict, history: str, action_policy=None) -> str:
    """PolicyExecutor._build_prompt. action_policy: a GameBoyRL action policy (default single)."""
    policy = action_policy or ACTION_POLICIES["single"]()
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
    """A GameBoyRL executor arm on a GameBoyPlayEnv (see the module docstring)."""

    def __init__(self, *, arm: str = DEFAULT_ARM, history_k: int = DEFAULT_HISTORY_K, game: str = "",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.arm = arm
        self.action_name, self.history_name = split_arm(arm)
        self.name = f"gameboy_{arm}"
        self.history_k = history_k
        self.game = game or getattr(self.env, "game", "")
        self._policy = ACTION_POLICIES[self.action_name]()
        self._history_policy = None
        self._last_reasoning: Optional[str] = None

    def reset_memory(self) -> None:
        # A fresh history policy per leg, as GameBoyRL builds one per executor (and resets it).
        self._history_policy = HISTORY_POLICIES[self.history_name](call=self._gb_call, game=self.game,
                                                                   history_k=self.history_k)
        self._history_policy.reset()
        self._last_reasoning = None

    # ------------------------------------------------------------ calls

    def _gb_call(self, tag: str, *, texts, images, max_new_tokens: Optional[int] = None):
        """Executor._vlm_call for the history policy's auxiliary calls: a list of prompts is one
        batched request, recorded one CallRecord per prompt, owning no steps."""
        if not isinstance(texts, list):
            return self.call(tag=tag, prompt=texts, images=images, max_new_tokens=max_new_tokens)
        out = self.vlm.infer(texts=texts, images=images, max_new_tokens=max_new_tokens or self.max_new_tokens,
                             temperature=self.temperature)
        responses = out["output"] if isinstance(out["output"], list) else [out["output"]] * len(texts)
        for i, (prompt, response) in enumerate(zip(texts, responses)):
            call_images = images[i] if i < len(images) and isinstance(images[i], list) else images
            self.report.calls.append(CallRecord(tag=tag, images=[EncodedImage.of(x) for x in call_images],
                                                response=response, prompt=prompt))
        return responses

    def _prompt(self, *, hint: Optional[str], error: Optional[str]) -> str:
        return build_step_prompt(task=self.task, hint=hint, steps_taken=len(self.report.env_steps), error=error,
                                 action_strings=self.env.action_strings(), history=self._history_policy.render(),
                                 action_policy=self._policy)

    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        """One deciding call. Decision.action_text holds the decision's actions joined by newlines
        (run() dispatches them one by one); invalid on a parse failure."""
        call_kwargs = {}
        if self._policy.max_new_tokens is not None:
            call_kwargs["max_new_tokens"] = self._policy.max_new_tokens
        response = self.call(tag=self._policy.tag, prompt=self._prompt(hint=hint, error=error),
                             images=[obs["frame"]], **call_kwargs)
        decision = self._policy.parse(response)
        if decision is None:
            log_warn(f"Invalid response recorded: \n{response}", parameters=self._parameters)
            return Decision(invalid="parse failure", error=self._policy.parse_error())
        self._last_reasoning = decision.reasoning or self._last_reasoning
        self._actions = list(decision.actions)
        return Decision(action_text="\n".join(decision.actions))

    # ------------------------------------------------------------ the loop

    def run(self, *, task: str, hint: Optional[str], max_steps: int, obs: dict, info: dict,
            allow_done_check: bool = False, env_name: str = "",
            max_consecutive_invalid: Optional[int] = MAX_CONSECUTIVE_INVALID, finish_through_env: bool = False,
            stop_on_model_error: bool = False) -> LegReport:
        """PolicyExecutor._execute. finish_through_env is moot (GameBoy has no finishing action:
        success is the test tracker terminating)."""
        limit = max_consecutive_invalid if max_consecutive_invalid is not None else float("inf")
        self.task = task
        self.report = LegReport(env_name=env_name, task=task, hint=hint, max_steps=max_steps,
                                initial_frame=EncodedImage.of(self.judge_frame(obs=obs, info=info)))
        self._current_call = None
        self.reset_memory()
        error: Optional[str] = None
        n_steps = 0
        consecutive_invalid = 0
        while n_steps < max_steps:
            try:
                decision = self.decide(obs=obs, info=info, error=error, hint=hint)
            except Exception as e:
                if not stop_on_model_error:
                    raise
                self.report.termination_reason, self.report.error = "model_error", f"{type(e).__name__}: {e}"[:500]
                return self.report
            deciding_call = self._current_call
            if decision.invalid is not None:
                deciding_call.decision = "invalid"
                self._record(InvalidRecord(response=deciding_call.response, reason=decision.invalid))
                error = decision.error
                n_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= limit:
                    self.report.termination_reason = "max_invalid"
                    return self.report
                continue
            deciding_call.decision = "step"
            outcome, obs, info, n_steps, consecutive_invalid, error = self._run_decision(
                obs=obs, info=info, n_steps=n_steps, max_steps=max_steps, consecutive_invalid=consecutive_invalid,
                limit=limit, hint=hint, allow_done_check=allow_done_check)
            if outcome is not None:
                self.report.termination_reason = outcome
                return self.report
        self.report.termination_reason = "max_steps"
        return self.report

    def _run_decision(self, *, obs, info, n_steps, max_steps, consecutive_invalid, limit, hint, allow_done_check):
        """PolicyExecutor._run_decision. Returns (outcome or None, obs, info, n_steps,
        consecutive_invalid, error)."""
        gb_steps, last_step = [], None
        error, aborted = None, False
        for action_str in self._actions:
            if n_steps >= max_steps:
                break
            obs_after, reward, terminated, truncated, info_after = self.env.step(action_str)
            if info_after.get("parsed_action") is None:
                # An action string the controller cannot parse never reaches the emulator.
                self._record(InvalidRecord(response=self._current_call.response, reason="unrecognised action"))
                error = self._policy.unknown_action_error(action_str)
                n_steps += 1
                consecutive_invalid += 1
                aborted = True
                break
            step = StepRecord(frame_before=EncodedImage.of(self.judge_frame(obs=obs, info=info)),
                              frame_after=EncodedImage.of(self.judge_frame(obs=obs_after, info=info_after)),
                              action_text=action_str, parsed_action=info_after.get("parsed_action"),
                              valid=bool(info_after["valid"]), error=info_after.get("error"), reward=float(reward),
                              extra={k: info_after[k] for k in ("action_name", "frame_changed", "action_success",
                                                                "low_level") if k in info_after})
            self._record(step)
            self.report.final_reward = float(reward)
            gb_steps.append(GBEnvironmentStepRecord(
                frame_before=info_after["core_frame_before"], frame_after=info_after["core_frame_after"],
                action_class=info_after["action_class"], kwargs=info_after["action_kwargs"],
                transition_states=info_after["transition_states"], action_success=info_after["action_success"],
                reward=reward, frame_changed=info_after["frame_changed"]))
            last_step = step
            obs, info = obs_after, info_after
            n_steps += 1
            consecutive_invalid = 0
            if terminated or truncated:
                self._history_policy.observe(gb_steps)
                return ("terminated" if terminated else "truncated"), obs, info, n_steps, consecutive_invalid, error
            # A failed high-level action ends a multi-action decision (low-level actions report
            # success 0 by convention, so they never do).
            if len(self._actions) > 1 and not info_after["low_level"] and info_after["action_success"] == 0:
                error = f"Action '{action_str}' failed (blocked or invalid). Re-plan."
                aborted = True
                break
        if gb_steps:
            self._history_policy.observe(gb_steps)
        if consecutive_invalid >= limit:
            return "max_invalid", obs, info, n_steps, consecutive_invalid, error
        if allow_done_check and last_step is not None and not aborted and self._done_check_due(n_steps=n_steps,
                                                                                               max_steps=max_steps):
            if self.done_check(step=last_step, hint=hint):
                return "agent_done", obs, info, n_steps, consecutive_invalid, error
        return None, obs, info, n_steps, consecutive_invalid, error

    def _done_check_due(self, *, n_steps: int, max_steps: int) -> bool:
        k = max(1, DONE_CHECK_EVERY_K_STEPS)
        if k == 1:
            return True
        return len(self.report.env_steps) % k == 0 or n_steps >= max_steps

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
