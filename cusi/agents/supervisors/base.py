"""
The supervisor contract: it runs executor legs on one episode and files them, with its own model calls, on a
SupervisorReport.
"""
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_error
from cusi.agents.executors.base import GUIDANCE_END, GUIDANCE_START
from cusi.agents.records import EncodedImage, per_prompt_token_counts
from cusi.agents.supervisors.report import LegEvent, SupervisorCall, SupervisorReport

#: Termination reasons that end the episode whatever the supervisor would do next.
EPISODE_ENDINGS = ("terminated", "truncated", "model_error")
SUPERVISOR_MAX_NEW_TOKENS = 5000


class Supervisor(ABC):
    """Runs executor legs on one episode; make_executor(previous_history=...) builds a fresh executor per leg.
    task None = exploring; guidance (practice only) is shown to the executor in every leg."""

    #: Mark the guidance (GUIDANCE markers, after the supervisor's own hint) for an executor that shows both in one
    #: hint block (GameBoy), so training data can cut the guidance alone. M3A and WebVoyager mark it themselves, and
    #: their supervisor never hints.
    MARK_GUIDANCE = False

    def __init__(self, *, task: str, guidance: Optional[str], env_name: str, env,
                 make_executor: Callable[..., Any], vlm, obs: dict, info: dict, max_steps: int,
                 parameters: Optional[dict] = None) -> None:
        self._task = task
        self._guidance = guidance
        self._env_name = env_name
        self._env = env
        self._make_executor = make_executor
        self._vlm = vlm
        self._max_steps = max_steps
        self._parameters = load_parameters(parameters)
        self.obs, self.info = obs, info
        self.executors: list = []
        #: The last leg's history(), carried into the next leg.
        self._carried_history = None
        self.report = SupervisorReport(task=task, supervisor_name=self.__class__.__name__, env=env_name,
                                       init_kwargs=self._run_config())
        self._evaluated = False

    def _run_config(self) -> dict:
        return {"supervisor_model": getattr(self._vlm, "model_name", None), "max_steps": self._max_steps}

    # ------------------------------------------------------------ recording

    def _vlm_call(self, stage: str, *, texts, images=None, max_new_tokens: int = SUPERVISOR_MAX_NEW_TOKENS) -> Any:
        """One supervisor call, or one batch when texts is a list (one record per prompt)."""
        t0 = time.time()
        inferred = self._vlm.infer(texts=texts, images=images, max_new_tokens=max_new_tokens, temperature=None)
        seconds = time.time() - t0
        result, meta = inferred["output"], inferred["meta"]
        images = images or []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            token_counts = per_prompt_token_counts(meta=meta, n_prompts=len(texts))
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                self.report.event_log.append(SupervisorCall(
                    stage=stage, images=[EncodedImage.of(i) for i in call_images], prompt=prompt, response=response,
                    input_tokens=token_counts[index][0], output_tokens=token_counts[index][1],
                    seconds=seconds if index == 0 else 0.0))
        else:
            self.report.event_log.append(SupervisorCall(
                stage=stage, images=[EncodedImage.of(i) for i in images], prompt=texts, response=result,
                input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"], seconds=seconds))
        return result

    # ------------------------------------------------------------ running

    def evaluate(self) -> dict:
        if self._evaluated:
            log_error(f"{self.__class__.__name__}.evaluate() called twice; build one supervisor per episode.",
                      parameters=self._parameters)
        self._evaluated = True
        extras = self._evaluate() or {}
        return {"report": self.report, **extras}

    @abstractmethod
    def _evaluate(self) -> Optional[dict]:
        raise NotImplementedError

    def call_executor(self, *, task: str, hint: Optional[str], max_steps: int, target: Optional[str] = None,
                      final: bool = True, self_terminate: bool = False):
        """Run one leg from the current state and file it -> its LegReport. The executor sees the guidance and then
        the supervisor's hint; self_terminate lets its done-check end the leg."""
        executor = self._make_executor(previous_history=self._carried_history)
        t0 = time.time()
        if self.MARK_GUIDANCE and self._guidance:
            shown = (hint or "") + f"{GUIDANCE_START}\nGuidance: {self._guidance}{GUIDANCE_END}"
        else:
            shown = "\n".join(h for h in (self._guidance, hint) if h) or None
        report = executor.run(task=task, hint=shown, max_steps=max_steps, obs=self.obs, info=self.info,
                              self_terminate=self_terminate)
        self.obs, self.info = executor.final_obs, executor.final_info
        self._carried_history = executor.history()
        self.executors.append(executor)
        self.report.event_log.append(LegEvent(report=report, target=target, final=final,
                                              self_terminate=self_terminate, seconds=time.time() - t0))
        return report
