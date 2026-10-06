"""
The supervisor arms: which supervisor wraps the executor, and with what settings.
"""
from cusi.agents.supervisors.dummy import DummySupervisor
from cusi.agents.supervisors.info_subgoal import InfoSubgoalSupervisor
from cusi.agents.supervisors.revising import RevisingSupervisor
from cusi.agents.supervisors.subgoal import SubgoalSupervisor

SUPERVISORS = {
    "baseline": DummySupervisor,
    "revision": RevisingSupervisor,
    "subgoal": SubgoalSupervisor,
    "info_subgoal_retrieval": InfoSubgoalSupervisor,
    "info_subgoal_parametric": InfoSubgoalSupervisor,
}
SUPERVISOR_MAX_NEW_TOKENS = 5000
DEFAULT_MAX_LEG_STEPS = 10   # GameBoyRL uses 5
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_REPLANS = 2
DEFAULT_MAX_FRAMES_PER_SLICE = 8


def build_supervisor(*, name: str, task: str, env, domain, make_executor, obs: dict, info: dict, max_steps: int,
                     run_kwargs: dict, vlm, game: str = "", max_leg_steps: int = DEFAULT_MAX_LEG_STEPS,
                     documents=None, parameters: dict):
    """The supervisor for one episode, with GameBoyRL's settings for its arm."""
    common = dict(task=task, env=env, domain=domain, make_executor=make_executor, obs=obs, info=info,
                  max_steps=max_steps, run_kwargs=run_kwargs, game=game, max_new_tokens=SUPERVISOR_MAX_NEW_TOKENS,
                  parameters=parameters)
    if name == "baseline":
        return DummySupervisor(**common)
    common.update(vlm=vlm, max_leg_steps=max_leg_steps, max_frames_per_slice=DEFAULT_MAX_FRAMES_PER_SLICE)
    if name == "revision":
        return RevisingSupervisor(**common)
    common.update(max_attempts_per_target=DEFAULT_MAX_ATTEMPTS, max_replans=DEFAULT_MAX_REPLANS)
    if name == "subgoal":
        return SubgoalSupervisor(**common)
    return InfoSubgoalSupervisor(documents=documents, **common)
