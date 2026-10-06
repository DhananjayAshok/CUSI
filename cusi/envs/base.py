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

Also on every environment:
    env_description  one sentence naming what is operated ("an Android phone", ...),
                     for environment-agnostic prompts
    sample_action()  a random valid action text for the current state (start-state
                     perturbation in the practice pipeline)
    current_obs      the latest observation (set by reset, step and load_state)

Saved states (for search over states, e.g. pre-exploration):
    state_id = env.save_state()       # snapshot the current state; returns a unique id
    env.load_state(state_id=state_id) # return to it; then read env.current_obs
    env.delete_state(state_id=state_id)
load_state starts a new episode from the saved state (step count 0, as after reset).
Ids are only valid on the env instance that saved them; close() deletes all of them.
A failed save, load or delete calls log_error.
"""
import hashlib
import string
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_error

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

    #: One sentence naming what is being operated, used by environment-agnostic prompts
    #: (task proposal, judging, guidance) in place of "a GameBoy game". Subclasses set it.
    env_description: str = "an interactive environment"

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
        self.current_obs: Optional[dict] = None
        self._saved_state_ids: set[str] = set()

    # gym API. step() takes the action positionally, as gym requires.
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Return to the scene's initial state. The scene itself is fixed at
        construction; `seed`/`options` only seed gym's RNG."""
        super().reset(seed=seed)
        self._episode_over = False
        obs, info = self._reset_impl()
        self._check_texts(obs=obs)
        self.current_obs = obs
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
        self.current_obs = obs
        return obs, reward, terminated, truncated, info

    # Saved states. Subclasses implement _save_state_impl / _load_state_impl / _delete_state_impl.
    def save_state(self) -> str:
        """Snapshot the current state; returns its unique id."""
        state_id = uuid.uuid4().hex[:16]
        self._save_state_impl(state_id=state_id)
        self._saved_state_ids.add(state_id)
        return state_id

    def load_state(self, *, state_id: str) -> None:
        """Return to a saved state and start a new episode there. The observation is in
        self.current_obs."""
        self._check_state_id(state_id=state_id)
        obs = self._load_state_impl(state_id=state_id)
        self._check_texts(obs=obs)
        self._episode_over = False
        self.current_obs = obs

    def delete_state(self, *, state_id: str) -> None:
        self._check_state_id(state_id=state_id)
        self._delete_state_impl(state_id=state_id)
        self._saved_state_ids.discard(state_id)

    def delete_all_states(self) -> None:
        for state_id in list(self._saved_state_ids):
            self.delete_state(state_id=state_id)

    @property
    def saved_state_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._saved_state_ids))

    def _check_state_id(self, *, state_id: str) -> None:
        if state_id not in self._saved_state_ids:
            log_error(f"No saved state {state_id!r} on this env", parameters=self._parameters)

    def _save_state_impl(self, *, state_id: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement save_state")

    def _load_state_impl(self, *, state_id: str) -> dict:
        """Restore the saved state; return the observation there."""
        raise NotImplementedError(f"{type(self).__name__} does not implement load_state")

    def _delete_state_impl(self, *, state_id: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement delete_state")

    def _check_texts(self, *, obs: dict) -> None:
        if set(obs["texts"]) != set(self.text_keys):
            log_error(f"obs['texts'] keys {sorted(obs['texts'])} != declared text_keys "
                      f"{sorted(self.text_keys)}", parameters=self._parameters)

    def sample_action(self) -> str:
        """A random valid action text for the current state (used to perturb start
        states). Uses gym's RNG (self.np_random), so reset(seed=...) makes it reproducible."""
        raise NotImplementedError(f"{type(self).__name__} does not implement sample_action")

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
