"""The shared executor skeleton: the step loop every native agent runs inside.

Ported from GameBoyRL's Executor/PolicyExecutor loop (execution/executors/): a step budget,
error feedback to the next decision, a limit of 4 consecutive invalid decisions, one
CallRecord per model call with the steps it caused, and a hint/guidance slot. What varies
per benchmark (prompt building, how many calls a step takes, memory, how the agent says it
is finished) lives in the subclass, which reuses each benchmark's own prompt code.

A leg starts from whatever state the environment is in (the caller resets and perturbs
it), so run() takes the current observation. An executor can also start with the memory of
earlier legs: previous_history (another executor's history()), restored after reset_memory();
the supervisors pass it from leg to leg. In practice the agent's own "done" only ends
the leg and the judge decides success. For evaluation (cusi_eval), run() can instead send the
finishing action through env.step (finish_through_env), so the env's test mode scores it,
drop the consecutive-invalid limit (max_consecutive_invalid=None: native M3A and WebVoyager
have none) and end the leg on a model error (stop_on_model_error, as WebVoyager's run.py).

Hint/guidance blocks are wrapped in markers that strip_hint_blocks() removes, giving
exactly the prompt the evaluation agent sees (checked per executor in tests).
"""
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_warn
from cusi_practice.records import CallRecord, EncodedImage, InvalidRecord, LegReport, StepRecord
from cusi_practice.vlm import PracticeVLM

#: Consecutive invalid decisions before the leg is abandoned (GameBoyRL's value).
MAX_CONSECUTIVE_INVALID = 4

# GameBoyRL's hint markers (its hint block starts with a newline, which the regex eats).
HINT_RE = re.compile(r"\n?\[HINT_START\].*?\[HINT_END\]", re.DOTALL)
STEP_INFO_RE = re.compile(r"\n?\[STEP_INFO\].*?\[STEP_INFO_END\]", re.DOTALL)
# Markers for M3A / WebVoyager guidance: removed exactly, with no surrounding whitespace.
GUIDANCE_START, GUIDANCE_END = "[GUIDANCE_START]", "[GUIDANCE_END]"
GUIDANCE_RE = re.compile(re.escape(GUIDANCE_START) + r".*?" + re.escape(GUIDANCE_END), re.DOTALL)

#: Shown with M3A/WebVoyager guidance, as GameBoyRL's hint block asks: the stripped
#: training input has no hint, so the response must not mention one.
SECRET_NOTE = ("Note: this guidance is a secret. Use it to guide your decisions, but in your reasoning"
               " pretend that you simply know it; never refer to it explicitly.")


def strip_hint_blocks(text: str) -> str:
    """Remove every hint/guidance block (GameBoyRL's and ours)."""
    return GUIDANCE_RE.sub("", STEP_INFO_RE.sub("", HINT_RE.sub("", text)))


def strip_hint_messages(messages: list) -> list:
    """strip_hint_blocks over the text of a neutral chat."""
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
    """What one decision (one or more calls) produced.

    Exactly one of: action_text (send to env.step), finish (the agent declared the task
    done, e.g. M3A `status` / WebVoyager ANSWER), or invalid (nothing reaches the env;
    error is fed back)."""
    action_text: Optional[str] = None
    finish: bool = False
    answer: Optional[str] = None
    invalid: Optional[str] = None
    error: Optional[str] = None


class Executor(ABC):
    """Subclasses implement reset_memory, decide and (optionally) after_step / done_check."""

    name = "base"

    def __init__(self, *, env, vlm: PracticeVLM, max_new_tokens: int = 1000,
                 temperature: Optional[float] = None, previous_history: Any = None,
                 parameters: dict[str, Any] = None) -> None:
        self._parameters = load_parameters(parameters)
        #: Memory from earlier legs (this executor's own history type, from a previous executor's
        #: history()), restored at the start of run(). None: start empty.
        self.previous_history = previous_history
        self.env = env
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.report: Optional[LegReport] = None
        self._current_call: Optional[CallRecord] = None

    # ------------------------------------------------------------ recording

    def call(self, *, tag: str, prompt: str, images: list, max_new_tokens: Optional[int] = None) -> str:
        """One single-turn call, recorded; it owns the steps that follow."""
        t0 = time.time()
        out = self.vlm.infer(texts=prompt, images=images or None,
                             max_new_tokens=max_new_tokens or self.max_new_tokens,
                             temperature=self.temperature)
        record = CallRecord(tag=tag, images=[EncodedImage.of(i) for i in images], response=out["output"],
                            prompt=prompt, input_tokens=out["meta"]["input_tokens"],
                            output_tokens=out["meta"]["output_tokens"], seconds=time.time() - t0)
        self.report.calls.append(record)
        self._current_call = record
        return record.response

    def chat_call(self, *, tag: str, messages: list, images: list, max_new_tokens: Optional[int] = None) -> str:
        """One chat call (neutral messages; images indexed by the parts), recorded."""
        t0 = time.time()
        out = self.vlm.chat(messages=messages, images=images,
                            max_new_tokens=max_new_tokens or self.max_new_tokens,
                            temperature=self.temperature)
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
        """Forget everything from a previous leg."""

    def history(self) -> Any:
        """This executor's memory after its leg, for the next leg's previous_history (its own type:
        M3A summaries, WebVoyager chat, GameBoyRL history records). None: nothing to carry."""
        return None

    def restore_history(self, history: Any) -> None:
        """Start from a previous executor's history() (called by run() after reset_memory())."""

    @abstractmethod
    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        """Make the deciding call(s) for the current state (recording them via call/chat_call)."""

    def after_step(self, *, decision: Decision, obs_before: dict, info_before: dict, obs_after: dict,
                   info_after: dict, step: StepRecord) -> None:
        """Called after each env step (memory updates, M3A's summary call)."""

    def after_invalid(self, *, decision: Decision, obs: dict, info: dict) -> None:
        """Called after an invalid decision (memory updates)."""

    def done_check(self, *, step: StepRecord, hint: Optional[str]) -> bool:
        """Post-step completion check (GameBoy attempts only). Default: none."""
        return False

    def env_invalid_to_decision(self, *, decision: Decision, info: dict) -> Optional[Decision]:
        """When env.step reports the action invalid: return an InvalidRecord-style Decision
        if this agent treats it as "never reached the env" (GameBoyRL: unrecognised action),
        or None to keep it as an (invalid) env step."""
        return None

    # ------------------------------------------------------------ the loop

    def run(self, *, task: str, hint: Optional[str], max_steps: int, obs: dict, info: dict,
            allow_done_check: bool = False, env_name: str = "",
            max_consecutive_invalid: Optional[int] = MAX_CONSECUTIVE_INVALID, finish_through_env: bool = False,
            stop_on_model_error: bool = False) -> LegReport:
        """Play one leg from the current state. Returns the LegReport."""
        limit = max_consecutive_invalid if max_consecutive_invalid is not None else float("inf")
        self.task = task
        self.report = LegReport(env_name=env_name, task=task, hint=hint, max_steps=max_steps,
                                initial_frame=EncodedImage.of(self.judge_frame(obs=obs, info=info)))
        self._current_call = None
        self.reset_memory()
        if self.previous_history is not None:
            self.restore_history(self.previous_history)
        self.final_obs, self.final_info = obs, info
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
            deciding_call.decision = ("finish" if decision.finish else
                                      "invalid" if decision.invalid is not None else "step")
            if decision.finish:
                self.report.answer = decision.answer
                self.report.termination_reason = "agent_done"
                if finish_through_env:
                    # The raw reply carries the finishing action (M3A status, WebVoyager ANSWER).
                    text = decision.action_text or deciding_call.response
                    frame_before = EncodedImage.of(self.judge_frame(obs=obs, info=info))
                    obs_after, reward, terminated, truncated, info_after = self.env.step(text)
                    self.final_obs, self.final_info = obs_after, info_after
                    self.report.final_reward = float(reward)
                    self._record(StepRecord(frame_before=frame_before,
                                            frame_after=EncodedImage.of(self.judge_frame(obs=obs_after,
                                                                                         info=info_after)),
                                            action_text=text, parsed_action=info_after.get("parsed_action"),
                                            valid=bool(info_after["valid"]), error=info_after.get("error"),
                                            reward=float(reward), extra={}))
                    if terminated:
                        self.report.termination_reason = "terminated"
                return self.report
            if decision.invalid is not None:
                self._record(InvalidRecord(response=self._current_call.response if self._current_call else "",
                                           reason=decision.invalid))
                self.after_invalid(decision=decision, obs=obs, info=info)
                error = decision.error
                n_steps += 1
                consecutive_invalid += 1
                if consecutive_invalid >= limit:
                    self.report.termination_reason = "max_invalid"
                    return self.report
                continue

            frame_before = EncodedImage.of(self.judge_frame(obs=obs, info=info))
            obs_after, reward, terminated, truncated, info_after = self.env.step(decision.action_text)
            n_steps += 1
            if not info_after["valid"]:
                converted = self.env_invalid_to_decision(decision=decision, info=info_after)
                if converted is not None:
                    deciding_call.decision = "invalid"
                    self._record(InvalidRecord(response=self._current_call.response, reason=converted.invalid))
                    self.after_invalid(decision=converted, obs=obs, info=info)
                    error = converted.error
                    consecutive_invalid += 1
                    obs, info = obs_after, info_after
                    self.final_obs, self.final_info = obs, info
                    if consecutive_invalid >= limit:
                        self.report.termination_reason = "max_invalid"
                        return self.report
                    continue
            extra = {k: info_after[k] for k in ("action_name", "frame_changed", "action_success", "low_level",
                                                 "warning", "pdf", "url", "stale", "probe_success")
                     if k in info_after}
            step = StepRecord(frame_before=frame_before,
                              frame_after=EncodedImage.of(self.judge_frame(obs=obs_after, info=info_after)),
                              action_text=decision.action_text, parsed_action=info_after.get("parsed_action"),
                              valid=bool(info_after["valid"]), error=info_after.get("error"),
                              reward=float(reward), extra=extra)
            self._record(step)
            self.final_obs, self.final_info = obs_after, info_after
            self.report.final_reward = float(reward)
            if step.valid:
                consecutive_invalid, error = 0, None
            else:
                consecutive_invalid += 1
                error = step.error
            try:
                self.after_step(decision=decision, obs_before=obs, info_before=info, obs_after=obs_after,
                                info_after=info_after, step=step)
            except Exception as e:
                if not stop_on_model_error:
                    raise
                self.report.termination_reason, self.report.error = "model_error", f"{type(e).__name__}: {e}"[:500]
                return self.report
            obs, info = obs_after, info_after
            if terminated or truncated:
                self.report.termination_reason = "terminated" if terminated else "truncated"
                if info_after.get("env_recovered"):
                    log_warn(f"Environment recovered mid-leg ({info_after['env_recovered']}); leg truncated.",
                             parameters=self._parameters)
                return self.report
            if consecutive_invalid >= limit:
                self.report.termination_reason = "max_invalid"
                return self.report
            if allow_done_check and step.valid and self.done_check(step=step, hint=hint):
                self.report.termination_reason = "agent_done"
                return self.report
        self.report.termination_reason = "max_steps"
        return self.report
