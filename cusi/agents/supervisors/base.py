"""The supervisor contract: GameBoyRL's Supervisor (execution/supervisors/base.py) over our envs
and executors.

A supervisor owns the env, the task, a factory for fresh executors and (if it reasons) a model.
call_executor() runs one leg: a fresh executor object started from the env's current observation,
given the previous leg's memory (its history(): M3A summaries, WebVoyager chat, GameBoyRL
history records) as previous_history, so memory is never reset within an episode (a change from
GameBoyRL, which starts every leg with empty memory). It files the leg's LegReport.

Leg endings (GameBoyRL's allow_self_termination):
    GameBoy        allow_self_termination turns on the executor's done-check, as native.
    Android / Web  the agent's own finishing action (M3A status, WebVoyager ANSWER). On a leg that
                   may end itself it ends the leg without reaching the env (the supervisor's judge
                   decides); on any other leg (the final target, baseline) it goes through env.step,
                   so the benchmark scores it and the episode ends there, as the native harness.
"""
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_error
from cusi.agents.records import EncodedImage, per_prompt_token_counts
from cusi.agents.supervisors.prompts import SupervisorDomain, prompts_for
from cusi.agents.supervisors.report import LegEvent, SupervisorCall, SupervisorReport

#: Termination reasons that end the episode whatever the supervisor would do next. "model_error"
#: is ours (WebVoyager's run.py ends the task on a model error); GameBoy never produces it.
EPISODE_ENDINGS = ("terminated", "truncated", "model_error")


class Supervisor(ABC):
    """:param make_executor: () -> a fresh Executor for one leg.
    :param run_kwargs: extra Executor.run arguments (env_name, max_consecutive_invalid, ...).
    :param obs, info: the env's observation after reset (legs chain from the latest one).
    :param game: GameBoy's game name, filled into GameBoyRL's [GAME]; unused elsewhere.
    :param vlm: the supervisor's model (a AgentVLM); None for a supervisor that makes no calls.
    :param max_new_tokens: token budget for every supervisor call (GameBoyRL: 5000)."""

    def __init__(self, *, task: str, env, domain: SupervisorDomain, make_executor: Callable[[], Any],
                 obs: dict, info: dict, max_steps: int, run_kwargs: Optional[dict] = None, game: str = "",
                 vlm=None, max_new_tokens: int = 5000, temperature: Optional[float] = None,
                 parameters: Optional[dict] = None) -> None:
        self._task = task
        self._env = env
        self._domain = domain
        self._prompts = prompts_for(domain=domain)
        self._make_executor = make_executor
        self._run_kwargs = dict(run_kwargs or {})
        self._game = game
        self._max_steps = max_steps
        self._vlm_instance = vlm
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._parameters = load_parameters(parameters)
        self.obs, self.info = obs, info
        #: Every leg's executor, in order (WebVoyager's native artefacts need each leg's chat and
        #: screenshots).
        self.executors: list = []
        #: The executor memory carried from leg to leg: the last leg's history(), passed to the next
        #: leg's executor as previous_history. Never reset within an episode (replans included).
        self._carried_history = None
        self.report = SupervisorReport(task=task, supervisor_name=self.__class__.__name__, env=domain.env,
                                       init_kwargs=self._run_config())
        self._evaluated = False

    def _run_config(self) -> dict:
        return {"supervisor_model": getattr(self._vlm_instance, "model_name", None),
                "max_new_tokens": self._max_new_tokens, "max_steps": self._max_steps,
                "temperature": self._temperature}

    # ------------------------------------------------------------ recording

    @property
    def _vlm(self):
        if self._vlm_instance is None:
            log_error(f"{self.__class__.__name__} tried to make a model call but has no supervisor model.",
                      parameters=self._parameters)
        return self._vlm_instance

    def _vlm_call(self, stage: str, **kwargs: Any) -> Any:
        """One supervisor call (a prompt, or a batch of prompts), recorded on the report."""
        result, records = self._vlm_infer(stage, **kwargs)
        self.report.event_log.extend(records)
        return result

    def _vlm_infer(self, stage: str, **kwargs: Any) -> tuple:
        """The call, returning (result, records) without filing them (for thread pools)."""
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        t0 = time.time()
        inferred = self._vlm.infer(texts=kwargs["texts"], images=kwargs.get("images"),
                                   max_new_tokens=kwargs["max_new_tokens"], temperature=self._temperature)
        seconds = time.time() - t0
        result, meta = inferred["output"], inferred["meta"]
        texts = kwargs.get("texts")
        images = kwargs.get("images") or []
        records = []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            token_counts = per_prompt_token_counts(meta=meta, n_prompts=len(texts))
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                records.append(SupervisorCall(stage=stage, images=[EncodedImage.of(i) for i in call_images],
                                              prompt=prompt, response=response, input_tokens=token_counts[index][0],
                                              output_tokens=token_counts[index][1],
                                              seconds=seconds if index == 0 else 0.0))
        else:
            records.append(SupervisorCall(stage=stage, images=[EncodedImage.of(i) for i in images], prompt=texts,
                                          response=result, input_tokens=meta["input_tokens"],
                                          output_tokens=meta["output_tokens"], seconds=seconds))
        return result, records

    def _vlm_caller(self, stage: str):
        def call(**kwargs: Any) -> Any:
            return self._vlm_call(stage, **kwargs)
        return call

    # ------------------------------------------------------------ state

    def current_frame(self):
        """The screen now: GameBoy's core frame (as GameBoyRL), else the unlabelled screenshot."""
        if self._domain.env == "gameboy":
            return self._env._env.get_info()["core"]["current_frame"]
        return self.info.get("raw_frame", self.obs["frame"])

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

    def call_executor(self, task: str, *, hint: Optional[str] = None, allow_self_termination: bool = False,
                      max_steps: Optional[int] = None, target: Optional[str] = None, final: bool = True) -> Any:
        """Run one leg from the current state and file it. See the module docstring for the endings."""
        executor = self._make_executor(previous_history=self._carried_history)
        t0 = time.time()
        report = executor.run(task=task, hint=hint, max_steps=self._max_steps if max_steps is None else max_steps,
                              obs=self.obs, info=self.info, allow_done_check=allow_self_termination,
                              finish_through_env=not allow_self_termination, **self._run_kwargs)
        self.obs, self.info = executor.final_obs, executor.final_info
        self._carried_history = executor.history()
        self.executors.append(executor)
        self.report.event_log.append(LegEvent(report=report, target=target, final=final,
                                              self_terminate=allow_self_termination, seconds=time.time() - t0))
        return self.process_executor_return(report)

    @abstractmethod
    def process_executor_return(self, report) -> Any:
        raise NotImplementedError
