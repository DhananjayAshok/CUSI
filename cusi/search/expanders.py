"""Exploration policies for expanding a node (plans/skill_discovery.md §1.4 D). All act through the
env's text actions; the search loop steps the env.

    expander.start(env=env)                                         # at the start of an expansion
    action, meta = expander.next_action(env=env, obs=obs, info=info)
    expander.observe(action=action, changed=<did the state change>) # after the step

RandomExpander   env.sample_action(). With avoid_noops, an action that left the state unchanged
                 (by the shared similarity, decided by the search loop) is not re-sampled while
                 the state stays the same (a no-op filter without domain state).
VLMExplorer      one goal-free policy for all envs: env_description + the env's action text +
                 the current frame and flattened text + the last few actions; the reply goes to
                 env.step as is (the env parses it). Minimal; to be replaced by the standardised
                 agent. Curiosity PPO as an expander is on hold.
"""
from typing import Optional
from cusi.state.canvas import frame_for_embedding
from cusi.state.text import element_lines

MAX_TEXT_LINES = 80
MAX_TEXT_CHARS = 4000


def flatten_texts(*, texts: dict) -> str:
    """The provisional text format: element lines, truncated for prompts."""
    lines = element_lines(texts=texts)[:MAX_TEXT_LINES]
    out = "\n".join(lines)
    return out[:MAX_TEXT_CHARS] if out else "(no text)"


class RandomExpander:
    name = "random"

    def __init__(self, *, avoid_noops: bool = True, max_resample: int = 10) -> None:
        self.avoid_noops, self.max_resample = avoid_noops, max_resample
        self._noops: set = set()

    def start(self, *, env) -> None:
        self._noops = set()

    def next_action(self, *, env, obs: dict, info: dict) -> tuple:
        action = env.sample_action()
        tries = 1
        while self.avoid_noops and action in self._noops and tries < self.max_resample:
            action = env.sample_action()
            tries += 1
        return action, {"resamples": tries - 1}

    def observe(self, *, action: str, changed: bool, valid: bool = True, error: Optional[str] = None) -> None:
        if changed:
            self._noops = set()
        elif self.avoid_noops:
            self._noops.add(action)


EXPLORE_PROMPT = """You are operating {env_description}. No task is given: your job is to explore. \
Do something you have not done yet in this session, so that you reach screens, menus, pages or \
features you have not seen before.

Available actions and their exact format:
{actions}

Text currently on the screen:
{texts}

Your last actions (oldest first):
{history}
{error}
Look at the screenshot. Reply with exactly one action, written in the format given above."""


class VLMExplorer:
    name = "vlm"

    def __init__(self, *, vlm, history_k: int = 5, max_new_tokens: int = 256, temperature: float = 1.0,
                 use_raw_frame: bool = False) -> None:
        """use_raw_frame=False: the VLM sees obs["frame"] (set-of-mark labels on Web/Android, which
        the action format refers to)."""
        self.vlm, self.history_k = vlm, history_k
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.use_raw_frame = use_raw_frame
        self.history: list = []
        self._last_error: Optional[str] = None

    def start(self, *, env) -> None:
        self.history = []
        self._last_error = None

    def prompt(self, *, env, obs: dict) -> str:
        history = "\n".join(f"- {a}" for a in self.history[-self.history_k:]) or "(none)"
        error = f"\nYour previous action failed: {self._last_error}\n" if self._last_error else ""
        return EXPLORE_PROMPT.format(env_description=env.env_description, actions=obs.get("actions", ""),
                                     texts=flatten_texts(texts=obs.get("texts", {})), history=history, error=error)

    def next_action(self, *, env, obs: dict, info: dict) -> tuple:
        frame = frame_for_embedding(obs=obs, info=info) if self.use_raw_frame else obs["frame"]
        text = self.prompt(env=env, obs=obs)
        out = self.vlm.infer(texts=text, images=[frame], max_new_tokens=self.max_new_tokens,
                             temperature=self.temperature)
        return str(out["output"]), {"prompt_chars": len(text)}

    def observe(self, *, action: str, changed: bool, valid: bool = True, error: Optional[str] = None) -> None:
        line = action.strip().splitlines()[-1][:200] if action.strip() else "(empty)"
        self.history.append(line if valid else f"{line}  [failed]")
        self._last_error = None if valid else error
