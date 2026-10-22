"""
Curiosity modules: a cusi.state scorer plus the PPO-side archive lifetime and reward shaping.
"""
from typing import Any, Optional
import numpy as np
from cusi.utils.log_handling import log_info
from cusi.state import NoveltyArchive, StateRecord
from cusi.state.scorers.base import Scorer


class RunningStd:
    def __init__(self, *, eps: float = 1e-8) -> None:
        self.mean, self.var, self.count, self.eps = 0.0, 1.0, 1e-4, eps

    def update(self, x: float) -> None:
        delta = x - self.mean
        total = self.count + 1
        self.mean += delta / total
        self.var = (self.var * self.count + delta ** 2 * self.count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + self.eps))


class CuriosityModule:
    def __init__(self, *, scorer: Scorer, archive: NoveltyArchive, buffer_load_path: Optional[str] = None,
                 save_path: Optional[str] = None, invalid_action_penalty: float = 0.1,
                 normalize_curiosity_reward: bool = False, gamma: float = 0.99,
                 parameters: dict[str, Any] = None) -> None:
        self.scorer, self.archive = scorer, archive
        self.save_path, self.buffer_load_path = save_path, buffer_load_path
        self.invalid_action_penalty = invalid_action_penalty
        self.normalize, self.gamma = normalize_curiosity_reward, gamma
        self._parameters = parameters
        if buffer_load_path is not None:
            archive.load_from(directory=buffer_load_path)
            log_info(f"Curiosity prior: {len(archive)} states from {buffer_load_path}", parameters=parameters)
        self._prior = archive.copy()
        # The frame part is scored (and first_add applies) unless the scorer is region-only.
        self._uses_frame = scorer.name == "embedding" or (scorer.name == "combination" and scorer.region_alpha < 1.0)
        self.first_add = False
        self._ret, self._rms = 0.0, RunningStd()

    def reset(self) -> None:
        """Restore the archive to its prior; call every episode."""
        self.archive.restore(self._prior)
        self.first_add = False

    def get_reward(self, *, prev: Optional[StateRecord], action: Any, next: StateRecord, valid: bool = True,
                   done: bool = False) -> dict:
        was_empty = len(self.archive) == 0
        value, components = self.scorer.score(prev=prev, action=action, next=next, archive=self.archive, add=True,
                                              seed_frame=self.first_add)
        if self._uses_frame:
            self.first_add = was_empty
        penalty = 0.0 if valid else self.invalid_action_penalty
        total = value - penalty
        if self.normalize:
            self._ret = self._ret * self.gamma + total
            self._rms.update(self._ret)
            total = total / self._rms.std
            if done:
                self._ret = 0.0
        return {"total": float(total), "novelty": float(value), "penalty": float(penalty),
                "frame": float(components.get("frame", 0.0)), "region": float(components.get("region", 0.0)),
                **{k: float(v) for k, v in components.items() if k not in ("frame", "region")}}

    def save(self) -> None:
        if self.save_path is not None and len(self.archive):
            self.archive.save(directory=self.save_path, merge=True)


