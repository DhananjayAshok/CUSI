"""
The search's environments: one per scene (its root), built on first use.
"""
from typing import Callable


class SceneEnvs:
    """env key (scene) -> live env. exclusive=True keeps one scene live at a time (Android: one emulator holds one
    scene). Switching away suspends the live env when it can be (AndroidPlayEnv.suspend: its saved states stay on the
    emulator, and resume() brings it back at its scene's start), else closes it (a rebuilt env has no saved states)."""

    def __init__(self, *, make: Callable, scenes: list, exclusive: bool = False) -> None:
        self._make, self._scenes, self.exclusive = make, list(scenes), exclusive
        self._envs: dict = {}
        self._suspended: dict = {}
        self._descriptions: dict = {}

    def description(self, key: str) -> str:
        """The scene's env_description, without building (or, when exclusive, switching to) its env."""
        if key not in self._descriptions:
            self._descriptions[key] = self[key].env_description
        return self._descriptions[key]

    def __getitem__(self, key: str):
        if key not in self._scenes:
            raise KeyError(f"No scene {key!r}; scenes are {self._scenes}")
        if key not in self._envs:
            if self.exclusive:
                self._set_aside()
            if key in self._suspended:
                env = self._suspended.pop(key)
                env.resume()
                self._envs[key] = env
            else:
                self._envs[key] = self._make(scene=key)
                self._descriptions[key] = self._envs[key].env_description
        return self._envs[key]

    def _set_aside(self) -> None:
        for key, env in list(self._envs.items()):
            if hasattr(env, "suspend"):
                env.suspend()
                self._suspended[key] = env
            else:
                try:
                    env.close()
                except Exception:
                    pass
        self._envs = {}

    def keys(self) -> list:
        return list(self._scenes)

    def items(self):
        for key in self._scenes:
            yield key, self[key]

    def close(self) -> None:
        for env in [*self._envs.values(), *self._suspended.values()]:
            try:
                env.close()
            except Exception:
                pass
        self._envs, self._suspended = {}, {}
