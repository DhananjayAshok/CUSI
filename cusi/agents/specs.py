"""The three environments as the pipeline uses them: scenes, executors, prompt domains.

    spec = ENV_SPECS["android"]
    env = spec.make_env(scene=spec.scenes["contacts"], worker=0, parameters=parameters)
    executor = spec.make_executor(env=env, vlm=vlm, parameters=parameters)

Environments always run in free_play: the proposed task lives in the executor's prompt.
Modules are imported lazily so a process only loads the benchmark it runs (GameBoyRL's and
WebVoyager's top-level `utils` modules cannot share a process).

Android workers use one emulator each, started beforehand by scripts/android_emulator.sh on
console port 5554 + 2*worker and gRPC port 8554 + worker (--snapshots true).
"""
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from cusi.practice.prompts import ANDROID_DOMAIN, WEB_DOMAIN, Domain, gameboy_domain

ENV_NAMES = ("gameboy", "android", "web")
GAMEBOY_GAME = "pokemon_red"


@dataclass(frozen=True)
class Scene:
    name: str
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EnvSpec:
    name: str
    domain: Domain
    scenes: dict
    max_steps: int               # step budget per leg (attempt and practice)
    max_workers: int             # concurrent environment instances
    make_env: Callable
    make_executor: Callable
    #: Calls tagged with these become dataset rows.
    dataset_tags: tuple = ("action",)


def _gameboy_env(*, scene: Scene, worker: int, parameters: dict):
    from cusi.envs.gameboy import GameBoyPlayEnv
    return GameBoyPlayEnv(game=GAMEBOY_GAME, mode="free_play", parameters=parameters, **scene.kwargs)


def _android_env(*, scene: Scene, worker: int, parameters: dict):
    from cusi.envs.android_world import AndroidEmulator, AndroidPlayEnv
    port, grpc = 5554 + 2 * worker, 8554 + worker
    emulator = AndroidEmulator(port=port, grpc_port=grpc, snapshots=True, parameters=parameters)
    return AndroidPlayEnv(mode="free_play", console_port=port, grpc_port=grpc, emulator=emulator,
                          reset_mode="snapshot", parameters=parameters, **scene.kwargs)


def _web_env(*, scene: Scene, worker: int, parameters: dict):
    from cusi.envs.webvoyager import WebVoyagerPlayEnv
    return WebVoyagerPlayEnv(mode="free_play", fast_waits=True, parameters=parameters, **scene.kwargs)


def _gameboy_executor(**kwargs):
    from cusi.agents.executors.gameboy import GameBoyExecutor
    return GameBoyExecutor(**kwargs)


def _m3a_executor(**kwargs):
    from cusi.agents.executors.m3a import M3AExecutor
    return M3AExecutor(**kwargs)


def _web_executor(**kwargs):
    from cusi.agents.executors.webvoyager import WebVoyagerExecutor
    return WebVoyagerExecutor(**kwargs)


ENV_SPECS = {
    # GameBoyRL's attempt/practice budget (--max_steps 50).
    "gameboy": EnvSpec(
        name="gameboy", domain=gameboy_domain(game=GAMEBOY_GAME),
        scenes={"viridian": Scene("viridian", {"init_state": "location_viridian_city_starting_charmander"}),
                "pewter": Scene("pewter", {"init_state": "location_pewter_city_starting_charmander"})},
        max_steps=50, max_workers=8, make_env=_gameboy_env, make_executor=_gameboy_executor),
    # AndroidWorld's evaluation budget is 10 x task complexity; 20 covers complexity <= 2.
    "android": EnvSpec(
        name="android", domain=ANDROID_DOMAIN,
        # start_app: AndroidWorld tasks start on the home screen, which would make the two
        # scenes look identical; each scene opens its task's app as its last step.
        scenes={"contacts": Scene("contacts", {"task": "ContactsAddContact", "start_app": "Contacts"}),
                "settings": Scene("settings", {"task": "SystemWifiTurnOff", "start_app": "Settings"})},
        max_steps=20, max_workers=1, make_env=_android_env, make_executor=_m3a_executor,
        dataset_tags=("action", "summary")),
    # WebVoyager's run.sh --max_iter 15.
    "web": EnvSpec(
        name="web", domain=WEB_DOMAIN,
        scenes={"arxiv": Scene("arxiv", {"url": "https://arxiv.org/"}),
                "github": Scene("github", {"url": "https://github.com/"})},
        max_steps=15, max_workers=2, make_env=_web_env, make_executor=_web_executor),
}


def proposal_context(*, env, obs: dict) -> tuple:
    """(action space text, texts block) for the propose prompt."""
    if hasattr(env, "action_strings"):
        # GameBoyRL's get_first_frame_and_actions: one "- <description>" line per action.
        actions = "".join(f"- {s}\n" for s in env.action_strings(return_all=True).values())
    else:
        actions = obs["actions"]
    texts = "\n\n".join(v for v in obs["texts"].values() if v)
    return actions, texts


class EnvPool:
    """One live environment per (worker, scene), reused across legs (Android scenes take a
    minute to build). Not thread-safe per worker: each worker thread uses its own index."""

    def __init__(self, *, spec: EnvSpec, parameters: dict[str, Any]) -> None:
        self.spec = spec
        self._parameters = parameters
        self._envs: dict = {}
        self._lock = threading.Lock()

    def get(self, *, worker: int, scene: str):
        key = (worker, scene)
        with self._lock:
            env = self._envs.get(key)
            # One env per worker at a time: an Android emulator holds one scene snapshot.
            stale = [(k, e) for k, e in self._envs.items() if k[0] == worker and k != key]
            for k, _ in stale:
                del self._envs[k]
        for _, e in stale:
            e.close()
        if env is None:
            env = self.spec.make_env(scene=self.spec.scenes[scene], worker=worker, parameters=self._parameters)
            with self._lock:
                self._envs[key] = env
        return env

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs = {}
