# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding, not part of the kept code.
# Strip it by deleting cusi_search/toy/, run_preexplore_toy.py, slurm/preexplore_toy.sh,
# scripts/slurm/preexplore_toy.sh, tests/*toy* and the TOY block in skill_discovery.md §1.5.
# Nothing outside those files imports this.
"""Toy env-per-scene: one env instance per root scene (cusi_practice.envs scenes), keyed by scene
name; a node's saved state only works on the env that made it. Android scenes would share one
emulator, so only one Android scene per search here.
"""
from cusi_practice.envs import ENV_SPECS


def make_envs(*, env_name: str, scenes: list, worker: int = 0, parameters: dict) -> dict:
    spec = ENV_SPECS[env_name]
    if env_name == "android" and len(scenes) > 1:
        raise ValueError("toy: one Android scene per search (scenes share one emulator)")
    envs = {}
    for scene in scenes:
        if scene not in spec.scenes:
            raise ValueError(f"unknown {env_name} scene {scene!r}; known: {sorted(spec.scenes)}")
        envs[scene] = spec.make_env(scene=spec.scenes[scene], worker=worker, parameters=parameters)
    return envs


def close_envs(*, envs: dict) -> None:
    for env in envs.values():
        try:
            env.close()
        except Exception:
            pass
