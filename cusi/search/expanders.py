"""
Exploration policies that expand a node through the env's text actions: random and a goal-free VLM.
"""
from typing import Optional
import numpy as np
from cusi.search.context import action_line, path_lines, tried_lines
from cusi.envs.text_boxes import render_boxes
from cusi.state.canvas import frame_for_embedding

MAX_TEXT_LINES = 80
MAX_TEXT_CHARS = 4000


def screen_text(*, boxes: list) -> str:
    """The screen's text boxes ('<role>: <text>' lines), truncated for prompts."""
    out = render_boxes(boxes=boxes, max_lines=MAX_TEXT_LINES, max_chars=MAX_TEXT_CHARS)
    return out if out else "(no text)"


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

    def observe(self, *, action: str, changed: bool, valid: bool = True, error: Optional[str] = None) -> None:
        if changed:
            self._noops = set()
        elif self.avoid_noops:
            self._noops.add(action)


EXPLORE_PROMPT = """You are operating {env_description}. No task is given: your job is to explore. \
Do something that has not been done before, so that you reach screens, menus, pages or features \
that have not been seen yet.

Available actions and their exact format:
{actions}

How the screen you started this exploration from was reached from the start screen (oldest first):
{path}

Earlier explorations that started from that same screen (do something different from these):
{tried}

Text currently on the screen:
{texts}
{elements}
Your actions so far in this exploration (oldest first):
{history}
{error}
Look at the screenshot. Reply with exactly one action, written in the format given above."""


def _bullets(lines: list, *, empty: str) -> str:
    return "\n".join(f"- {line}" for line in lines) or empty


class VLMExplorer:
    name = "vlm"

    def __init__(self, *, vlm, history_k: int = 5, max_new_tokens: int = 256, temperature: float = 1.0,
                 use_raw_frame: bool = False, max_path_actions: int = 20, max_tries: int = 5,
                 actions_per_try: int = 5) -> None:
        """use_raw_frame=False shows the set-of-mark frame, whose labels the action format refers to. The prompt
        also carries the node's root->node actions (the last max_path_actions) and the earlier expansions from it
        (the last max_tries, actions_per_try actions each, and where each ended)."""
        self.vlm, self.history_k = vlm, history_k
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.use_raw_frame = use_raw_frame
        self.max_path_actions, self.max_tries, self.actions_per_try = max_path_actions, max_tries, actions_per_try
        self.history: list = []
        self._last_error: Optional[str] = None
        self._path, self._tried = "", ""
        self._context: dict = {}

    def start(self, *, env, iteration: int, tree=None, node=None, expansion_id: Optional[int] = None,
              search=None) -> None:
        """tree/node: the node being expanded; expansion_id: this expansion (earlier ones are listed as tried)."""
        self.history = []
        self._last_error = None
        self._context = {}
        if tree is None or node is None:
            self._path, self._tried = "(unknown)", "(unknown)"
            return
        self._path = _bullets(path_lines(tree=tree, node_id=node.id, max_actions=self.max_path_actions),
                              empty="(none: it is the start screen)")
        before = expansion_id if expansion_id is not None else len(tree.expansions)
        self._tried = _bullets(tried_lines(tree=tree, node_id=node.id, before_expansion=before,
                                           max_tries=self.max_tries, actions_per_try=self.actions_per_try),
                               empty="(none: this is the first exploration from it)")
        tried = [e.id for e in tree.expansions if e.node_id == node.id and e.id < before and e.steps]
        self._context = {"path_actions": len(tree.path_steps(node.id)), "tried_ids": tried[-self.max_tries:]}

    def info(self) -> dict:
        """Expansion.expander_info: the model and the tree context its prompts were given."""
        return {"model_name": getattr(self.vlm, "model_name", None), "context": dict(self._context)}

    def prompt(self, *, env, obs: dict) -> str:
        history = "\n".join(f"- {a}" for a in self.history[-self.history_k:]) or "(none)"
        error = f"\nYour previous action failed: {self._last_error}\n" if self._last_error else ""
        # The element list the env's own agent is shown (Web: WebVoyager's numbered elements), so labels can be
        # chosen from what exists.
        elements = f"\n{env.elements_intro}\n{obs['elements']}\n" if obs.get("elements") is not None else ""
        return EXPLORE_PROMPT.format(env_description=env.env_description, actions=obs.get("actions", ""),
                                     path=self._path, tried=self._tried,
                                     texts=screen_text(boxes=obs.get("text_boxes", [])), elements=elements,
                                     history=history, error=error)

    def next_action(self, *, env, obs: dict, info: dict) -> tuple:
        frame = frame_for_embedding(obs=obs, info=info) if self.use_raw_frame else obs["frame"]
        text = self.prompt(env=env, obs=obs)
        out = self.vlm.infer(texts=text, images=[frame], max_new_tokens=self.max_new_tokens,
                             temperature=self.temperature)
        return str(out["output"]), {"prompt_chars": len(text)}

    def observe(self, *, action: str, changed: bool, valid: bool = True, error: Optional[str] = None) -> None:
        self.history.append(action_line(action=action, valid=valid))
        self._last_error = None if valid else error
