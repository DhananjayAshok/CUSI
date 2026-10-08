"""
The common contract for CUSI's text-action environments: a gym.Env whose observation is a frame plus
named texts, and whose action is the model's text. "test" mode is the benchmark; "free_play" has no goal.
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

# gym's Text space needs a charset; this documents the content rather than constraining it.
_TEXT_CHARSET = string.printable
_MAX_TEXT = 200_000


def scene_hash(*, parts: tuple) -> str:
    """Opaque, stable scene id (never the task name, which would leak the goal in free play)."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


class TextActionEnv(gym.Env, ABC):
    """Subclasses implement _reset_impl and _step_impl."""

    metadata = {"render_modes": []}

    #: What is being operated, for environment-agnostic prompts. Subclasses set it.
    env_description: str = "an interactive environment"

    def __init__(self, *, mode: str, frame_shape: tuple, text_keys: tuple[str, ...],
                 parameters: dict[str, Any] = None) -> None:
        """text_keys: the exact keys of every obs["texts"] (may be empty)."""
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
        """Return to the initial state; the scene is fixed at construction, seed only seeds gym's RNG."""
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
        """Snapshot the current state; the id is valid only on this env and is deleted by close()."""
        state_id = uuid.uuid4().hex[:16]
        self._save_state_impl(state_id=state_id)
        self._saved_state_ids.add(state_id)
        return state_id

    def load_state(self, *, state_id: str) -> None:
        """Start a new episode from a saved state; read the observation from self.current_obs."""
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

    def export_state(self, *, state_id: str, directory: str) -> str:
        """Copy a saved state into a file in directory, which outlives this env; returns the file's name."""
        self._check_state_id(state_id=state_id)
        return self._export_state_impl(state_id=state_id, directory=directory)

    def import_state(self, *, path: str) -> str:
        """Register a file from export_state (any instance of this env's scene) as a saved state; returns its id."""
        state_id = uuid.uuid4().hex[:16]
        self._import_state_impl(state_id=state_id, path=path)
        self._saved_state_ids.add(state_id)
        return state_id

    @property
    def saved_state_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._saved_state_ids))

    def _check_state_id(self, *, state_id: str) -> None:
        if state_id not in self._saved_state_ids:
            log_error(f"No saved state {state_id!r} on this env", parameters=self._parameters)

    def _save_state_impl(self, *, state_id: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement save_state")

    def _load_state_impl(self, *, state_id: str) -> dict:
        raise NotImplementedError(f"{type(self).__name__} does not implement load_state")

    def _export_state_impl(self, *, state_id: str, directory: str) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not implement export_state")

    def _import_state_impl(self, *, state_id: str, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement import_state")

    def _delete_state_impl(self, *, state_id: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement delete_state")

    def _check_texts(self, *, obs: dict) -> None:
        if set(obs["texts"]) != set(self.text_keys):
            log_error(f"obs['texts'] keys {sorted(obs['texts'])} != declared text_keys "
                      f"{sorted(self.text_keys)}", parameters=self._parameters)

    @property
    def raw_frame(self):
        """The current screen without set-of-mark labels (what info["raw_frame"] carries), after any reset, step
        or load_state. Envs whose obs frame is labelled override this."""
        return None if self.current_obs is None else self.current_obs["frame"]

    def sample_action(self, *, rng=None) -> str:
        """A random valid action text for the current state, drawn from rng (default: self.np_random)."""
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
