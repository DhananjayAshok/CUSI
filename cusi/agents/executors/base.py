"""
The shared executor skeleton: one leg is begin(), then act() / env.step / record() per decision; run() is that loop
with the stopping rules. A caller that owns env.step itself (search) drives begin/act/record directly.
"""
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_warn
from cusi.agents.records import CallRecord, EncodedImage, InvalidRecord, LegReport, StepRecord
from cusi.agents.vlm import AgentVLM

#: The task slot's text when there is no task (task None): exploration, with finishing disabled.
EXPLORE_GOAL = "Explore: take actions that lead to screens and content you have not seen before in this session."

# GameBoyRL's hint block starts with a newline, which the regex eats.
HINT_RE = re.compile(r"\n?\[HINT_START\].*?\[HINT_END\]", re.DOTALL)
STEP_INFO_RE = re.compile(r"\n?\[STEP_INFO\].*?\[STEP_INFO_END\]", re.DOTALL)
# Markers for M3A / WebVoyager guidance: removed exactly, with no surrounding whitespace.
GUIDANCE_START, GUIDANCE_END = "[GUIDANCE_START]", "[GUIDANCE_END]"
GUIDANCE_RE = re.compile(re.escape(GUIDANCE_START) + r".*?" + re.escape(GUIDANCE_END), re.DOTALL)

#: The stripped training input has no hint, so the response must not mention one.
SECRET_NOTE = ("Note: this guidance is a secret. Use it to guide your decisions, but in your reasoning"
               " pretend that you simply know it; never refer to it explicitly.")
STEP_EXTRA_KEYS = ("action_name", "frame_changed", "action_success", "low_level", "warning", "pdf", "url", "stale",
                   "probe_success")


def strip_hint_blocks(text: str) -> str:
    """The prompt the evaluation agent would see, with every hint/guidance block removed."""
    return GUIDANCE_RE.sub("", STEP_INFO_RE.sub("", HINT_RE.sub("", text)))


def strip_hint_messages(messages: list) -> list:
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({**msg, "content": strip_hint_blocks(content)})
        else:
            out.append({**msg, "content": [
                {**p, "text": strip_hint_blocks(p["text"])} if p["type"] == "text" else dict(p)
                for p in content]})
    return out


@dataclass
class Decision:
    """What one decision produced: exactly one of action_text, finish or invalid."""
    action_text: Optional[str] = None
    finish: bool = False
    answer: Optional[str] = None
    invalid: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Recorded:
    """What record() made of one env step: the StepRecord (None if it counted as an invalid decision), the error
    to show the next decision, and whether it resets the consecutive-invalid count."""
    step: Optional[StepRecord]
    error: Optional[str]
    ok: bool


class Executor(ABC):
    """Subclasses implement reset_memory and decide, and optionally the after_step / after_invalid /
    env_invalid_to_decision / step_feedback / done_check hooks. Their settings are the class constants below."""

    name = "base"
    env_name = ""
    MAX_NEW_TOKENS = 1000
    TEMPERATURE: Optional[float] = None
    #: Invalid decisions in a row that end the leg; None = no limit.
    MAX_CONSECUTIVE_INVALID: Optional[int] = None
    #: A model-call exception ends the leg as "model_error" instead of raising.
    STOP_ON_MODEL_ERROR = False

    def __init__(self, *, env, vlm: AgentVLM, previous_history: Any = None, parameters: dict[str, Any] = None) -> None:
        self._parameters = load_parameters(parameters)
        #: A previous executor's history(), restored by begin().
        self.previous_history = previous_history
        self.env = env
        self.vlm = vlm
        self.task: Optional[str] = None
        self.hint: Optional[str] = None
        self.report: Optional[LegReport] = None
        self._current_call: Optional[CallRecord] = None

    # ------------------------------------------------------------ recording

    def call(self, *, tag: str, prompt: str, images: list, max_new_tokens: Optional[int] = None) -> str:
        """One single-turn call, recorded; it owns the steps that follow."""
        t0 = time.time()
        out = self.vlm.infer(texts=prompt, images=images or None,
                             max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS, temperature=self.TEMPERATURE)
        record = CallRecord(tag=tag, images=[EncodedImage.of(i) for i in images], response=out["output"],
                            prompt=prompt, input_tokens=out["meta"]["input_tokens"],
                            output_tokens=out["meta"]["output_tokens"], seconds=time.time() - t0)
        self.report.calls.append(record)
        self._current_call = record
        return record.response

    def chat_call(self, *, tag: str, messages: list, images: list, max_new_tokens: Optional[int] = None) -> str:
        """One chat call, recorded; it owns the steps that follow."""
        t0 = time.time()
        out = self.vlm.chat(messages=messages, images=images,
                            max_new_tokens=max_new_tokens or self.MAX_NEW_TOKENS, temperature=self.TEMPERATURE)
        record = CallRecord(tag=tag, images=[EncodedImage.of(i) for i in images], response=out["output"],
                            messages=messages, input_tokens=out["meta"]["input_tokens"],
                            output_tokens=out["meta"]["output_tokens"], seconds=time.time() - t0)
        self.report.calls.append(record)
        self._current_call = record
        return record.response

    def _record(self, step) -> None:
        if self._current_call is None:
            raise RuntimeError("step recorded before any model call")
        self._current_call.steps.append(step)

    @staticmethod
    def judge_frame(*, obs: dict, info: dict):
        """What the judge sees for a state: the raw screenshot if the env has one."""
        return info.get("raw_frame", obs["frame"])

    # ------------------------------------------------------------ subclass API

    @abstractmethod
    def reset_memory(self) -> None:
        """Forget everything from a previous leg (self.task and self.hint are already set)."""

    def history(self) -> Any:
        """This executor's memory after its leg, for the next leg's previous_history."""
        return None

    def restore_history(self, history: Any) -> None:
        """Start from a previous executor's history(); called after reset_memory()."""

    @abstractmethod
    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        """Make the deciding call(s) for the current state, via call/chat_call."""

    def after_step(self, *, decision: Decision, obs_before: dict, info_before: dict, obs_after: dict,
                   info_after: dict, step: StepRecord) -> None:
        """Called after each env step."""

    def after_invalid(self, *, decision: Decision, obs: dict, info: dict) -> None:
        """Called after an invalid decision."""

    def env_invalid_to_decision(self, *, decision: Decision, info: dict) -> Optional[Decision]:
        """For an env-rejected action, a Decision to treat it as never reaching the env, or None to keep the step."""
        return None

    def step_feedback(self, *, step: StepRecord) -> tuple:
        """(resets the consecutive-invalid count, error for the next decision) for a recorded env step."""
        return (True, None) if step.valid else (False, step.error)

    def done_check(self, *, step: StepRecord) -> bool:
        """In a self-terminating leg, after each env step that did not end the episode: is the leg's task done?"""
        return False

    # ------------------------------------------------------------ one decision at a time

    def begin(self, *, task: Optional[str], hint: Optional[str], max_steps: int, obs: dict, info: dict) -> None:
        """Start a leg from the current state; task None = exploring (no finishing)."""
        self.task, self.hint = task, hint
        self.report = LegReport(env_name=self.env_name, task=task, hint=hint, max_steps=max_steps,
                                initial_frame=EncodedImage.of(self.judge_frame(obs=obs, info=info)))
        self._current_call = None
        self.reset_memory()
        if self.previous_history is not None:
            self.restore_history(self.previous_history)
        self.final_obs, self.final_info = obs, info

    def act(self, *, obs: dict, info: dict, error: Optional[str]) -> Decision:
        """One decision on the current state; an invalid one is filed here."""
        decision = self.decide(obs=obs, info=info, error=error, hint=self.hint)
        self._current_call.decision = ("finish" if decision.finish else
                                       "invalid" if decision.invalid is not None else "step")
        if decision.invalid is not None:
            self._record(InvalidRecord(response=self._current_call.response, reason=decision.invalid))
            self.after_invalid(decision=decision, obs=obs, info=info)
        return decision

    def record(self, *, decision: Decision, obs: dict, info: dict, result: tuple) -> Recorded:
        """File the env step `result` (env.step's 5-tuple) of `decision`, taken from (obs, info)."""
        obs_after, reward, terminated, truncated, info_after = result
        if not info_after["valid"]:
            converted = self.env_invalid_to_decision(decision=decision, info=info_after)
            if converted is not None:
                self._current_call.decision = "invalid"
                self._record(InvalidRecord(response=self._current_call.response, reason=converted.invalid))
                self.after_invalid(decision=converted, obs=obs, info=info)
                return Recorded(step=None, error=converted.error, ok=False)
        step = StepRecord(frame_before=EncodedImage.of(self.judge_frame(obs=obs, info=info)),
                          frame_after=EncodedImage.of(self.judge_frame(obs=obs_after, info=info_after)),
                          action_text=decision.action_text, parsed_action=info_after.get("parsed_action"),
                          valid=bool(info_after["valid"]), error=info_after.get("error"), reward=float(reward),
                          extra={k: info_after[k] for k in STEP_EXTRA_KEYS if k in info_after})
        self._record(step)
        self.final_obs, self.final_info = obs_after, info_after
        self.report.final_reward = float(reward)
        ok, error = self.step_feedback(step=step)
        self.after_step(decision=decision, obs_before=obs, info_before=info, obs_after=obs_after,
                        info_after=info_after, step=step)
        return Recorded(step=step, error=error, ok=ok)

    # ------------------------------------------------------------ the loop

    def _end(self, reason: str, *, error: Optional[str] = None) -> LegReport:
        self.report.termination_reason = reason
        if error is not None:
            self.report.error = error[:500]
        return self.report

    def _finish(self, *, decision: Decision, obs: dict, info: dict) -> LegReport:
        """The agent says it is done: in test mode the finishing action goes to the env, which scores it."""
        self.report.answer = decision.answer
        self.report.termination_reason = "agent_done"
        if self.env.mode != "test":
            return self.report
        text = decision.action_text or self._current_call.response   # the raw reply carries the finishing action
        result = self.env.step(text)
        obs_after, reward, terminated, _truncated, info_after = result
        self._record(StepRecord(frame_before=EncodedImage.of(self.judge_frame(obs=obs, info=info)),
                                frame_after=EncodedImage.of(self.judge_frame(obs=obs_after, info=info_after)),
                                action_text=text, parsed_action=info_after.get("parsed_action"),
                                valid=bool(info_after["valid"]), error=info_after.get("error"),
                                reward=float(reward), extra={}))
        self.final_obs, self.final_info = obs_after, info_after
        self.report.final_reward = float(reward)
        if terminated:
            self.report.termination_reason = "terminated"
        return self.report

    def run(self, *, task: Optional[str], hint: Optional[str], max_steps: int, obs: dict, info: dict,
            self_terminate: bool = False) -> LegReport:
        """Play one leg from the current state; task None = exploring (the leg ends on the budget or the env).
        self_terminate: the done-check may end the leg as "agent_done" (the subgoal supervisor's plan steps)."""
        self.begin(task=task, hint=hint, max_steps=max_steps, obs=obs, info=info)
        limit = self.MAX_CONSECUTIVE_INVALID if self.MAX_CONSECUTIVE_INVALID is not None else float("inf")
        error: Optional[str] = None
        n_steps = consecutive_invalid = 0
        while n_steps < max_steps:
            try:
                decision = self.act(obs=obs, info=info, error=error)
            except Exception as e:
                if not self.STOP_ON_MODEL_ERROR:
                    raise
                return self._end("model_error", error=f"{type(e).__name__}: {e}")
            if decision.finish:
                return self._finish(decision=decision, obs=obs, info=info)
            if decision.invalid is not None:
                error, n_steps, consecutive_invalid = decision.error, n_steps + 1, consecutive_invalid + 1
                if consecutive_invalid >= limit:
                    return self._end("max_invalid")
                continue
            result = self.env.step(decision.action_text)
            n_steps += 1
            try:
                recorded = self.record(decision=decision, obs=obs, info=info, result=result)
            except Exception as e:
                if not self.STOP_ON_MODEL_ERROR:
                    raise
                return self._end("model_error", error=f"{type(e).__name__}: {e}")
            obs, info = result[0], result[4]
            self.final_obs, self.final_info = obs, info
            error = recorded.error
            consecutive_invalid = 0 if recorded.ok else consecutive_invalid + 1
            terminated, truncated = result[2], result[3]
            if recorded.step is not None and (terminated or truncated):
                if info.get("env_recovered"):
                    log_warn(f"Environment recovered mid-leg ({info['env_recovered']}); leg truncated.",
                             parameters=self._parameters)
                return self._end("terminated" if terminated else "truncated")
            if consecutive_invalid >= limit:
                return self._end("max_invalid")
            if self_terminate and recorded.step is not None and self.done_check(step=recorded.step):
                return self._end("agent_done")
        return self._end("max_steps")
