"""Curiosity modules (curiosity_plan §3.3): thin wrappers around a cusi_state scorer, with the
PPO-side lifetime and shaping. GameBoyRL's interface on the outside.

    module = get_curiosity_module(curiosity_module="combinationbuffer", env_name=..., encoder=encoder,
                                  region_alpha=0.5, buffer_load_path=None, save_path=..., ...)
    module.reset()                                        # every episode: back to the prior
    r = module.get_reward(prev=prev_record, action=text, next=next_record, valid=info["valid"], done=done)
    r["total"], r["frame"], r["region"], r["novelty"], r["penalty"]
    module.save()                                         # merge this archive into save_path

Modules: embedbuffer = the `embedding` scorer, combinationbuffer = `combination`,
world_model = `world_model` (a loaded model; not trained during PPO).

Lifetime (not in cusi_state): reset() restores the archive to its prior (--buffer_load_path, or
empty) via copy()/restore(); first_add as GameBoyRL (after a reset into an empty archive, the
first two frames score 0: the first seeds it, the second is stored without scoring).
Shaping: --invalid_action_penalty (an action the env rejects gets -penalty on top of its novelty,
which is 0 for the unchanged screen) and --normalize_curiosity_reward (divide by a running std of
the discounted curiosity return, as gym's NormalizeReward does for env rewards; GameBoyRL keeps
the intrinsic reward raw).
"""
from typing import Any, Optional
import numpy as np
from cusi_utils.log_handling import log_info
from cusi_state import NoveltyArchive, StateRecord, build_archive, build_scorer
from cusi_state.scorers.base import Scorer

CURIOSITY_MODULES = {"embedbuffer": "embedding", "combinationbuffer": "combination", "world_model": "world_model"}


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


def get_curiosity_module(*, curiosity_module: str, env_name: str, encoder, region_alpha: float = 0.5,
                         world_model_load_path: Optional[str] = None, buffer_load_path: Optional[str] = None,
                         save_path: Optional[str] = None, invalid_action_penalty: float = 0.1,
                         normalize_curiosity_reward: bool = False, device: Optional[str] = None,
                         parameters: dict[str, Any] = None) -> CuriosityModule:
    if curiosity_module not in CURIOSITY_MODULES:
        raise ValueError(f"--curiosity_module must be one of {sorted(CURIOSITY_MODULES)}, got {curiosity_module!r}")
    scorer = build_scorer(novelty_scorer=CURIOSITY_MODULES[curiosity_module], env_name=env_name,
                          region_alpha=region_alpha, world_model_load_path=world_model_load_path, encoder=encoder,
                          device=device)
    return CuriosityModule(scorer=scorer, archive=build_archive(encoder=encoder), buffer_load_path=buffer_load_path,
                           save_path=save_path, invalid_action_penalty=invalid_action_penalty,
                           normalize_curiosity_reward=normalize_curiosity_reward, parameters=parameters)
