"""
Exploration policies that expand a node through the env's text actions: random, and the env's eval executor
pursuing objectives. An expander's observe() gets each env.step result; info["changed"] says whether the step changed
the screen.
"""
from typing import Optional
from urllib.parse import urlparse
import numpy as np
from cusi.search.context import explore_hint
from cusi.search.describe import normalise_phrase
from cusi.state.canvas import frame_for_embedding


class RandomExpander:
    """Random actions; with avoid_noops, an action that left the state unchanged is not re-sampled until it changes."""

    name = "random"

    def __init__(self, *, seed: int = 0, avoid_noops: bool = True, max_resample: int = 10) -> None:
        self.seed, self.avoid_noops, self.max_resample = seed, avoid_noops, max_resample
        self.rng = np.random.default_rng(seed)
        self._noops: set = set()

    def start(self, *, env, iteration: int, tree=None, node=None, expansion_id: Optional[int] = None,
              search=None) -> None:
        """Each expansion draws from a stream fixed by (seed, iteration), so a resumed search replays exactly.
        The tree context is not used."""
        self.rng = np.random.default_rng([self.seed, iteration])
        self._noops = set()

    def info(self) -> dict:
        """Expansion.expander_info for the expansion just started."""
        return {}

    def next_action(self, *, env, obs: dict, info: dict) -> tuple:
        action = env.sample_action(rng=self.rng)
        tries = 1
        while self.avoid_noops and action in self._noops and tries < self.max_resample:
            action = env.sample_action(rng=self.rng)
            tries += 1
        return action, {"resamples": tries - 1}

    def observe(self, *, action: str, result: tuple) -> None:
        if result[4].get("changed", True):
            self._noops = set()
        elif self.avoid_noops:
            self._noops.add(action)


#: The task an objective (a screen feature the scene has not discovered) is given to the executor as.
OBJECTIVE_TASK = 'Find and open "{feature}".'
#: Only when a screen yields no features at all.
FALLBACK_FEATURE = "a screen you have not seen before"
#: What the env is sent when a decision is invalid (or finishes twice running): it rejects it without acting, so the
#: decision uses a step.
NO_ACTION = ""
MAX_RESPONSE_CHARS = 2000


class ExecutorExpander:
    """VLM exploration by the env's eval executor (cusi.agents.agent), which search drives one decision at a time
    (begin / act / record). The walk's task is an objective: one of the screen's features (VLMFeatures; at the start,
    the node's) that the scene has not discovered and the walk has not pursued. The hint is the tree context
    (cusi.search.context.explore_hint). When the executor says the objective is done (its finish action, or the
    done-check every executor runs after each step; only GameBoy's makes a call), the current screen's features give
    the next objective and the executor is retasked, keeping its memory."""

    name = "vlm"

    def __init__(self, *, env_name: str, vlm, seed: int = 0, parameters: dict) -> None:
        self.env_name, self.vlm, self.seed, self._parameters = env_name, vlm, seed, parameters
        self.executor = None

    def start(self, *, env, iteration: int, tree, node, expansion_id: Optional[int] = None, search) -> None:
        from cusi.agents.agent import Agent
        self.env, self.search = env, search
        self.rng = np.random.default_rng([self.seed, iteration])
        self.discovered = search.discovered(env_key=node.env_key)
        self.objectives: list = []
        self._t, self._error, self._pending = 0, None, None
        site = getattr(env, "site_name", None) or (urlparse(env.start_url).netloc if getattr(env, "start_url", None)
                                                   else None)
        app = getattr(env, "scoped_app", None)
        before = expansion_id if expansion_id is not None else len(tree.expansions)
        self.hint = explore_hint(tree=tree, node_id=node.id, before_expansion=before, discovered=self.discovered,
                                 site=site, app=app)
        self.executor = Agent(env_name=self.env_name, env=env, vlm=self.vlm,
                              parameters=self._parameters).make_executor()
        self.executor.begin(task=self._objective(features=node.features), hint=self.hint, max_steps=search.k_steps,
                            obs=env.current_obs, info={"raw_frame": env.raw_frame})

    def _objective(self, *, features: Optional[list]) -> str:
        """The next objective's task, recorded."""
        pursued = {normalise_phrase(text=o["feature"]) for o in self.objectives}
        seen = pursued | {normalise_phrase(text=d) for d in self.discovered}
        features = list(features or [])
        open_ = ([f for f in features if normalise_phrase(text=f) not in seen]
                 or [f for f in features if normalise_phrase(text=f) not in pursued] or [FALLBACK_FEATURE])
        feature = open_[int(self.rng.integers(len(open_)))]
        self.objectives.append({"feature": feature, "step": self._t, "done": None})
        return OBJECTIVE_TASK.format(feature=feature)

    def _next_objective(self, *, obs: dict, info: dict, how: str) -> None:
        """The current objective is done (how: "finish" or "done_check"): the next one from this screen."""
        self.objectives[-1].update(done=how, done_step=self._t)
        f = self.search.features(env_description=self.env.env_description, frame=frame_for_embedding(obs=obs, info=info),
                                 text_boxes=obs.get("text_boxes", []))
        self.executor.retask(task=self._objective(features=f["features"]))

    def info(self) -> dict:
        calls: dict = {}
        for call in self.executor.report.calls:
            calls[call.tag] = calls.get(call.tag, 0) + 1
        return {"model_name": getattr(self.vlm, "model_name", None), "executor": self.executor.name,
                "hint": self.hint, "objectives": [dict(o) for o in self.objectives], "calls": calls}

    def next_action(self, *, env, obs: dict, info: dict) -> tuple:
        self._before, self._pending = (obs, info), None
        decision = self.executor.act(obs=obs, info=info, error=self._error)
        if decision.finish:
            self._next_objective(obs=obs, info=info, how="finish")
            decision = self.executor.act(obs=obs, info=info, error=None)
        meta = {"objective": len(self.objectives) - 1,
                "response": self.executor.report.calls[-1].response[:MAX_RESPONSE_CHARS]}
        if decision.finish or decision.invalid is not None:
            self._error = decision.error
            return NO_ACTION, {**meta, "decision": "finish" if decision.finish else f"invalid: {decision.invalid}"}
        self._pending = decision
        return decision.action_text, meta

    def observe(self, *, action: str, result: tuple) -> None:
        self._t += 1
        if self._pending is None:
            return
        obs, info = self._before
        recorded = self.executor.record(decision=self._pending, obs=obs, info=info, result=result)
        self._error = recorded.error
        if recorded.step is not None and not (result[2] or result[3]) and self.executor.done_check(step=recorded.step):
            self._next_objective(obs=result[0], info=result[4], how="done_check")
