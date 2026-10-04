"""The common contract for CUSI's text-action environments.

Every environment (AndroidWorld, WebVoyager, GameBoyWorlds) is a gym.Env whose
policy reads an image frame plus text and answers with a text action:

    env = SomeEnv(..., mode="test")          # the scene is fixed at construction
    obs, info = env.reset()                  # back to exactly that initial state
    obs, reward, terminated, truncated, info = env.step("<model output>")

Observation (a dict):
    frame    HxWx3 uint8 image (what the policy sees)
    texts    dict of named text channels, fixed per environment (its `text_keys`),
             e.g. {"ui_elements": <numbered list of on-screen elements>}
    actions  a description of the allowed actions and their text format
    goal     the task instruction in "test" mode; "" in "free_play" mode

Modes:
    test       the benchmark as evaluated: goal shown, the benchmark's finishing
               action ends the episode, reward = the benchmark's success signal.
    free_play  no goal, finishing actions are rejected (no-op), reward 0, and the
               episode never terminates (only truncates if max_steps is set).

info always holds:
    valid          False if the text did not parse or the action failed (a no-op)
    error          feedback string for an invalid/failed action, else None
    parsed_action  canonical dict of the parsed action (e.g. {"action_type":
                   "click", "index": 3}), or None
    step           steps taken since reset
    scene_id       opaque hash of the scene (never the task name, which would
                   leak most of the goal in free play)
"""
import hashlib
import string
from abc import ABC, abstractmethod
from typing import Any, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_error

MODES = ("test", "free_play")

# Text fields can hold arbitrary UI text; gym's Text space needs a charset, so this
# documents the expected content rather than constraining it.
_TEXT_CHARSET = string.printable
_MAX_TEXT = 200_000


def scene_hash(*, parts: tuple) -> str:
    """Opaque, stable id for a scene (benchmark, task, seed, ...)."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


class TextActionEnv(gym.Env, ABC):
    """Base class: a gym.Env with a dict (frame, texts, actions, goal) observation and a
    text action. Subclasses implement _reset_impl and _step_impl."""

    metadata = {"render_modes": []}

    def __init__(self, *, mode: str, frame_shape: tuple, text_keys: tuple[str, ...],
                 parameters: dict[str, Any] = None) -> None:
        """
        :param text_keys: Names of the text channels in obs["texts"]; every observation
            has exactly these keys. May be empty.
        """
        self._parameters = load_parameters(parameters)
        if mode not in MODES:
            log_error(f"mode must be one of {MODES}, got {mode!r}", parameters=self._parameters)
        self.mode = mode
        self.text_keys = tuple(text_keys)
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(low=0, high=255, shape=tuple(frame_shape), dtype=np.uint8),
            "texts": spaces.Dict({k: spaces.Text(max_length=_MAX_TEXT, charset=_TEXT_CHARSET)
                                  for k in self.text_keys}),
            "actions": spaces.Text(max_length=_MAX_TEXT, charset=_TEXT_CHARSET),
            "goal": spaces.Text(min_length=0, max_length=_MAX_TEXT, charset=_TEXT_CHARSET),
        })
        self.action_space = spaces.Text(max_length=_MAX_TEXT, charset=_TEXT_CHARSET)
        self._episode_over = False

    # gym API. step() takes the action positionally, as gym requires.
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Return to the scene's initial state. The scene itself is fixed at
        construction; `seed`/`options` only seed gym's RNG."""
        super().reset(seed=seed)
        self._episode_over = False
        obs, info = self._reset_impl()
        self._check_texts(obs=obs)
        return obs, info

    def step(self, action: str):
        return self.step_str(action)

    def step_str(self, action: str):
        if self._episode_over:
            log_error("step() called after the episode ended; call reset() first.",
                      parameters=self._parameters)
        obs, reward, terminated, truncated, info = self._step_impl(action)
        self._check_texts(obs=obs)
        self._episode_over = bool(terminated or truncated)
        return obs, reward, terminated, truncated, info

    def _check_texts(self, *, obs: dict) -> None:
        if set(obs["texts"]) != set(self.text_keys):
            log_error(f"obs['texts'] keys {sorted(obs['texts'])} != declared text_keys "
                      f"{sorted(self.text_keys)}", parameters=self._parameters)

    @abstractmethod
    def _reset_impl(self) -> tuple[dict, dict]:
        """Restore the initial state; return (obs, info)."""

    @abstractmethod
    def _step_impl(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Apply one text action; return (obs, reward, terminated, truncated, info)."""

    @staticmethod
    def make_info(*, valid: bool, error: Optional[str], parsed_action: Optional[dict],
                  step: int, scene_id: str, **extra) -> dict:
        info = {"valid": valid, "error": error, "parsed_action": parsed_action,
                "step": step, "scene_id": scene_id}
        info.update(extra)
        return info
