"""
Exploration policies that expand a node through the env's text actions: random, and the env's eval executor
pursuing objectives. An expander's observe() gets each env.step result; info["changed"] says whether the step changed
the screen.
"""
import json
import re
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


#: The typing explorer's task: the fields it may still type into (their index and label).
TYPING_TASK = ("Type into one of these text fields to find out what it leads to, with text this {where} would act on: "
               "{fields}. {actions}")
TYPING_ACTIONS = {"web": "Use only Type (it presses Enter after typing) and Scroll.",
                  "android": "Use only input_text, keyboard_enter (to submit what you typed), scroll, and "
                             '{"action_type": "hide_keyboard"} (to close the keyboard).'}
MSG_FIELD_CAPPED = ("That field has been typed into enough already, so the action was not performed. Type into one of "
                    "these instead: {fields}.")
_WEB_TYPE_RE = re.compile(r"Type\s*\[\s*(\d+)\s*\]\s*[;,]?\s*\[?(.*)", re.IGNORECASE | re.DOTALL)


def field_list(*, fields: list) -> str:
    """Fields as the executor is told them: [index] "label"."""
    return "; ".join(f'[{f["index"]}] "{f["label"]}"' if f["label"] else f'[{f["index"]}]' for f in fields) or "(none)"


def field_key(*, field: dict) -> str:
    """A text field's identity across screens: the same search box on every results page is one field. Web: page
    host, tag, type, name, placeholder, aria-label, label text, form action (not the page path, nor the element id,
    which some sites generate per page load). Android: package, resource name, class, hint, content description."""
    sig = field.get("signature") or {}
    keys = (("host", "tag", "type", "name", "placeholder", "aria_label", "label", "form_action") if "host" in sig
            else ("package", "resource_name", "class_name", "hint_text", "content_description"))
    return json.dumps([sig.get(k) or "" for k in keys])


class TypingExpander(ExecutorExpander):
    """VLM exploration by typing only (the env runs in action_mode "typing": Web Type and Scroll; Android input_text,
    keyboard_enter, scroll, hide_keyboard). It expands only nodes with a text field (node.typable, recorded by every
    expander) whose identity (field_key) has been typed into fewer than field_cap times in the scene; the executor's
    task lists those fields. Each typing step records the field it typed into (step extra typed_field). With no such
    node left, the search stops ("no typable field left")."""

    name = "vlm_typing"

    @property
    def exhausted_reason(self) -> str:
        """Why no node was left to expand: no open field anywhere, or open fields only on nodes that cannot be restored
        (Android replays that diverged)."""
        tree = self.search.tree if getattr(self, "search", None) is not None else None
        if tree is not None and any(not n.restorable and self.open_fields(fields=n.typable, search=self.search,
                                                                          env_key=n.env_key) for n in tree.nodes):
            return "no restorable node with a typable field left"
        return "no typable field left"

    def __init__(self, *, env_name: str, vlm, seed: int = 0, field_cap: int = 10, parameters: dict) -> None:
        super().__init__(env_name=env_name, vlm=vlm, seed=seed, parameters=parameters)
        self.field_cap = field_cap
        self._counts: dict = {}           # env_key -> {field key: times typed into}
        self._counted = None              # (expansions, steps of the last) the counts were made at

    def typed_counts(self, *, search, env_key: str) -> dict:
        """How often each field identity of the scene has been typed into: every recorded typing step of the tree,
        the expansion in progress included (recounted whenever a step has been added)."""
        expansions = search.tree.expansions
        at = (len(expansions), len(expansions[-1].steps) if expansions else 0)
        if at != self._counted:
            counts: dict = {}
            for e in expansions:
                for s in e.steps:
                    key = (s.extra.get(self.name) or {}).get("typed_field")
                    if key is not None:
                        scene = search.tree.nodes[e.node_id].env_key
                        counts.setdefault(scene, {})[key] = counts.get(scene, {}).get(key, 0) + 1
            self._counts, self._counted = counts, at
        return self._counts.get(env_key, {})

    def open_fields(self, *, fields: Optional[list], search, env_key: str) -> list:
        counts = self.typed_counts(search=search, env_key=env_key)
        return [f for f in (fields or []) if counts.get(field_key(field=f), 0) < self.field_cap]

    def eligible(self, *, node, search) -> bool:
        return bool(self.open_fields(fields=node.typable, search=search, env_key=node.env_key))

    def start(self, *, env, iteration: int, tree, node, expansion_id: Optional[int] = None, search) -> None:
        self._done, self._env_key = False, node.env_key
        super().start(env=env, iteration=iteration, tree=tree, node=node, expansion_id=expansion_id, search=search)

    def _objective(self, *, features: Optional[list]) -> str:
        """The task: type into one of the current screen's open fields (the features are not used)."""
        fields = self.open_fields(fields=self.env.typable_fields(), search=self.search, env_key=self._env_key)
        if not fields:
            self._done = True
        where = getattr(self.env, "site_name", None) or getattr(self.env, "scoped_app", None) or "screen"
        self.objectives.append({"fields": [f["index"] for f in fields], "step": self._t, "done": None})
        return TYPING_TASK.format(where=where, fields=field_list(fields=fields), actions=TYPING_ACTIONS[self.env_name])

    def _next_objective(self, *, obs: dict, info: dict, how: str) -> None:
        self.objectives[-1].update(done=how, done_step=self._t)
        self.executor.retask(task=self._objective(features=None))

    def done(self) -> bool:
        return self._done

    def after_step(self, *, obs: dict, info: dict, embedding) -> bool:
        """The walk goes on while the screen has a field to type into (else done() ends it)."""
        if not self.open_fields(fields=self.env.typable_fields(), search=self.search, env_key=self._env_key):
            self._done = True
        return True

    def _typed_field(self, *, action_text: str) -> tuple:
        """(field key, typed text) of a typing action on the current screen, or (None, None)."""
        index, text = None, None
        if self.env_name == "web":
            m = _WEB_TYPE_RE.search(action_text.splitlines()[-1] if action_text else "")
            if m:
                index, text = int(m.group(1)), m.group(2).strip().strip("[]")
        else:
            try:
                start = action_text.index("{")
                action = json.loads(action_text[start:action_text.rindex("}") + 1])
                if action.get("action_type") == "input_text":
                    index, text = action.get("index"), action.get("text")
            except (ValueError, json.JSONDecodeError):
                pass
        if index is None:
            return None, None
        field = next((f for f in self.env.typable_fields() if f["index"] == int(index)), None)
        return (field_key(field=field), text) if field is not None else (None, text)

    def next_action(self, *, env, obs: dict, info: dict) -> tuple:
        """A typing action into a field at the cap is not performed: the executor is told which fields are open
        (and with none open, the walk ends after this step)."""
        action, meta = super().next_action(env=env, obs=obs, info=info)
        if action:
            key, text = self._typed_field(action_text=action)
            if key is not None:
                if self.typed_counts(search=self.search, env_key=self._env_key).get(key, 0) >= self.field_cap:
                    fields = self.open_fields(fields=env.typable_fields(), search=self.search, env_key=self._env_key)
                    self._pending, self._error = None, MSG_FIELD_CAPPED.format(fields=field_list(fields=fields))
                    if not fields:
                        self._done = True
                    return NO_ACTION, {**meta, "decision": "capped field", "capped_field": key}
                meta = {**meta, "typed_field": key, "typed_text": text}
        return action, meta
