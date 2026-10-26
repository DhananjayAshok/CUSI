"""
The final agent of each env (decisions.md E8.0): one fixed executor and the supervisor that runs it. Eval and practice
build agents here (search's VLM exploration drives the bare executor); nothing about them is configurable except the
model client.

    agent = make_agent(env_name="android", env=env, vlm=vlm, parameters=parameters)
    supervisor, result = agent.run(task=goal, obs=obs, info=info, max_steps=budget)
    # result = {"report": SupervisorReport, **supervisor state}; supervisor.executors = one executor per leg
    leg = merged_leg(report=result["report"])     # the whole run as one LegReport (practice judges and trains on it)

guidance (practice's critique hints; never used in eval) is shown to the executor in every leg.
"""
from typing import Optional
from cusi.agents.records import LegReport

ENV_NAMES = ("gameboy", "android", "web")
#: Recorded in every run's config, so runs of the final agents never mix with earlier experiment runs.
AGENT_VERSION = "final-v1"


def _classes(env_name: str) -> tuple:
    """(executor class, supervisor class) for an env; imported lazily (GameBoyRL's and WebVoyager's `utils` modules
    cannot share a process)."""
    if env_name == "web":
        from cusi.agents.executors.webvoyager import WebVoyagerExecutor
        from cusi.agents.supervisors.direct import DirectSupervisor
        return WebVoyagerExecutor, DirectSupervisor
    if env_name == "android":
        from cusi.agents.executors.m3a import M3AExecutor
        from cusi.agents.supervisors.direct import DirectSupervisor
        return M3AExecutor, DirectSupervisor
    if env_name == "gameboy":
        from cusi.agents.executors.gameboy import GameBoyExecutor
        from cusi.agents.supervisors.subgoal import SubgoalSupervisor
        return GameBoyExecutor, SubgoalSupervisor
    raise ValueError(f"unknown env {env_name!r}; one of {ENV_NAMES}")


class Agent:
    """An env's executor under its supervisor; one supervisor (and one executor per leg) per run()."""

    def __init__(self, *, env_name: str, env, vlm, parameters: dict) -> None:
        self.env_name = env_name
        self.env = env
        self.vlm = vlm
        self._parameters = parameters
        self.executor_class, self.supervisor_class = _classes(env_name)

    def make_executor(self, *, previous_history=None):
        return self.executor_class(env=self.env, vlm=self.vlm, previous_history=previous_history,
                                   parameters=self._parameters)

    def run(self, *, task: str, obs: dict, info: dict, max_steps: int,
            guidance: Optional[str] = None) -> tuple:
        """One episode from the current state -> (supervisor, {"report": SupervisorReport, **supervisor state})."""
        supervisor = self.supervisor_class(task=task, guidance=guidance, env_name=self.env_name, env=self.env,
                                           make_executor=self.make_executor, vlm=self.vlm, obs=obs, info=info,
                                           max_steps=max_steps, parameters=self._parameters)
        return supervisor, supervisor.evaluate()


def make_agent(*, env_name: str, env, vlm, parameters: dict) -> Agent:
    return Agent(env_name=env_name, env=env, vlm=vlm, parameters=parameters)


def merged_leg(*, report) -> LegReport:
    """A run's legs as one LegReport: the run's task, every executor call in order, the first leg's start and the
    last leg's ending (a GameBoy plan-step leg's own task is the step, so the task comes from the run)."""
    legs = report.executor_reports
    first, last = legs[0], legs[-1]
    return LegReport(env_name=first.env_name, task=report.task, hint=first.hint,
                     max_steps=sum(leg.max_steps for leg in legs), calls=[c for leg in legs for c in leg.calls],
                     termination_reason=last.termination_reason, answer=last.answer,
                     initial_frame=first.initial_frame, final_reward=last.final_reward, error=last.error)
