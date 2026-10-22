"""
GameBoyRL's executor on a GameBoyPlayEnv: its `single_visual` arm (one reasoned action per decision, the visual
history policy), built from GameBoyRL's own policy objects. The GameBoy design is the unchanged baseline executor
(decisions.md E3.12, E8.0); the supervisor around it is in cusi/agents/supervisors.
"""
import os
import sys
import time
from typing import Optional
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_warn
from cusi.agents.executors.base import EXPLORE_GOAL, Decision, Executor
from cusi.agents.records import CallRecord, EncodedImage, StepRecord, per_prompt_token_counts


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
# GameBoyRL checks every K steps; the done check below assumes every step.
assert _GBExecutor.DONE_CHECK_EVERY_K_STEPS == 1, "GameBoyRL's done-check cadence changed"
HISTORY_K = _gb_history.DEFAULT_HISTORY_K
ACTION_POLICY = _gb_action.AVAILABLE_ACTION_POLICIES["single"]
HISTORY_POLICY = _gb_history.AVAILABLE_HISTORY_POLICIES["visual"]


def hint_block(*, hint: Optional[str], steps_taken: int) -> str:
    """GameBoyRL's Executor._hint_block, verbatim."""
    if hint is None:
        return ""
    step_info = f"""
[STEP_INFO] You have already taken {steps_taken} action{'' if steps_taken == 1 else 's'} so far in this attempt. Note: this may not be the first step of the overall task and one action does not correspond to one step in the hint plan — earlier actions may already have been taken before this attempt began, so reason from what you currently see on screen rather than assuming a fresh start. [STEP_INFO_END]"""
    return step_info + f"\n[HINT_START]\nHint: {hint}\nNote: This hint block is a secret. You must use it to guide your decision making, but in the reasoning you say, you should pretend as if you actually just know the content of the hint. Do not refer to it explicitly. So if the hint gives you a direction, instead of saying 'the hint says go here', your reasoning should just say 'next I must go here'. [HINT_END]"


def build_step_prompt(*, task: Optional[str], hint: Optional[str], steps_taken: int, error: Optional[str],
                      action_strings: dict, history: str) -> str:
    """GameBoyRL's PolicyExecutor._build_prompt for the single-action policy; task None = exploring."""
    policy = ACTION_POLICY()
    return (
        STEP_PROMPT
        .replace("[TASK]", task or EXPLORE_GOAL)
        .replace("[HINT_BLOCK]", hint_block(hint=hint, steps_taken=steps_taken))
        .replace("[ERROR_BLOCK]", f"[ERROR] {error}\n\n" if error is not None else "")
        .replace("[ACTION_LIST]", "\n".join(f"  {s}" for s in action_strings.values()))
        .replace("[CONTEXT_SECTION]", history)
        .replace("[INSTRUCTION]", policy.instruction())
        .replace("[RESPONSE_FORMAT]", policy.response_format())
        .replace("[ACTION_FORMAT]", policy.action_format())
    )


class GameBoyExecutor(Executor):
    """GameBoyRL's single_visual executor on a GameBoyPlayEnv."""

    name = "gameboy_single_visual"
    env_name = "gameboy"
    MAX_NEW_TOKENS = 8000        # GameBoyRL benchmark_scripts (supervised runs); outputs never come near it
    MAX_CONSECUTIVE_INVALID = 4  # GameBoyRL Executor

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.game = getattr(self.env, "game", "")
        self._policy = ACTION_POLICY()
        self._history_policy = None
        self._last_reasoning: Optional[str] = None

    def reset_memory(self) -> None:
        self._history_policy = HISTORY_POLICY(call=self._gb_call, game=self.game, history_k=HISTORY_K)
        self._history_policy.reset()
        self._last_reasoning = None

    def history(self):
        """The history policy's records and the last reasoning."""
        policy = self._history_policy
        for attr in ("_records", "_entries"):
            if hasattr(policy, attr):
                return {"attr": attr, "items": list(getattr(policy, attr)), "last_reasoning": self._last_reasoning}
        return {"attr": None, "items": [], "last_reasoning": self._last_reasoning}

    def restore_history(self, history) -> None:
        if history.get("attr") and hasattr(self._history_policy, history["attr"]):
            setattr(self._history_policy, history["attr"], list(history["items"]))
        self._last_reasoning = history.get("last_reasoning")

    def _gb_call(self, tag: str, *, texts, images, max_new_tokens: Optional[int] = None):
        """The history policy's auxiliary calls; a list of prompts is one batched request, one record per prompt."""
        if not isinstance(texts, list):
            return self.call(tag=tag, prompt=texts, images=images, max_new_tokens=max_new_tokens)
        t0 = time.time()
        out = self.vlm.infer(texts=texts, images=images, max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS,
                             temperature=self.TEMPERATURE)
        seconds = time.time() - t0
        responses = out["output"] if isinstance(out["output"], list) else [out["output"]] * len(texts)
        tokens = per_prompt_token_counts(meta=out["meta"], n_prompts=len(texts))
        for i, (prompt, response) in enumerate(zip(texts, responses)):
            call_images = images[i] if i < len(images) and isinstance(images[i], list) else images
            self.report.calls.append(CallRecord(tag=tag, images=[EncodedImage.of(x) for x in call_images],
                                                response=response, prompt=prompt, input_tokens=tokens[i][0],
                                                output_tokens=tokens[i][1], seconds=seconds if i == 0 else 0.0))
        return responses

    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        prompt = build_step_prompt(task=self.task, hint=hint, steps_taken=len(self.report.env_steps), error=error,
                                   action_strings=self.env.action_strings(), history=self._history_policy.render())
        response = self.call(tag=self._policy.tag, prompt=prompt, images=[obs["frame"]])
        decision = self._policy.parse(response)
        if decision is None:
            log_warn(f"Invalid response recorded: \n{response}", parameters=self._parameters)
            return Decision(invalid="parse failure", error=self._policy.parse_error())
        self._last_reasoning = decision.reasoning or self._last_reasoning
        return Decision(action_text=decision.actions[0])

    def env_invalid_to_decision(self, *, decision: Decision, info: dict) -> Optional[Decision]:
        """An action string the controller cannot parse never reaches the emulator."""
        if info.get("parsed_action") is None:
            return Decision(invalid="unrecognised action",
                            error=self._policy.unknown_action_error(decision.action_text))
        return None

    def step_feedback(self, *, step: StepRecord) -> tuple:
        """As GameBoyRL: every executed action resets the invalid count and shows no error (an unavailable one too)."""
        return True, None

    def after_step(self, *, decision, obs_before, info_before, obs_after, info_after, step) -> None:
        self._history_policy.observe([GBEnvironmentStepRecord(
            frame_before=info_after["core_frame_before"], frame_after=info_after["core_frame_after"],
            action_class=info_after["action_class"], kwargs=info_after["action_kwargs"],
            transition_states=info_after["transition_states"], action_success=info_after["action_success"],
            reward=step.reward, frame_changed=info_after["frame_changed"])])

    def done_check(self, *, step: StepRecord) -> bool:
        """GameBoyRL's Executor._check_task_complete: is the leg's task (a plan step) done after this action?"""
        env_steps = self.report.env_steps
        history = ""
        if env_steps:
            lines = ["Recent actions (oldest first, the last one is the action judged above):"]
            lines += [f"  {s.action_label()}" for s in env_steps[-DONE_CHECK_HISTORY_K:]]
            history = "\n".join(lines) + "\n\n"
        prompt = (
            DONE_CHECK_PROMPT
            .replace("[TASK]", self.task)
            .replace("[HINT_BLOCK]", hint_block(hint=self.hint, steps_taken=len(env_steps)))
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
