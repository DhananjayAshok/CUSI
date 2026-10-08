"""
The search's environments: one per scene (its root), built on first use.
"""
from typing import Callable


class SceneEnvs:
    """env key (scene) -> live env. exclusive=True keeps one scene live at a time (Android: one emulator holds one
    scene), closing the others; a rebuilt env has no saved states, so it suits restore="replay" only."""

    def __init__(self, *, make: Callable, scenes: list, exclusive: bool = False) -> None:
        self._make, self._scenes, self.exclusive = make, list(scenes), exclusive
        self._envs: dict = {}

    def __getitem__(self, key: str):
        if key not in self._scenes:
            raise KeyError(f"No scene {key!r}; scenes are {self._scenes}")
        if key not in self._envs:
            if self.exclusive:
                self.close()
            self._envs[key] = self._make(scene=key)
        return self._envs[key]

    def keys(self) -> list:
        return list(self._scenes)

    def items(self):
        for key in self._scenes:
            yield key, self[key]

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs = {}
