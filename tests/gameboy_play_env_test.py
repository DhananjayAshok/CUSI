"""
Checks GameBoyPlayEnv's observations, actions, resets and truncation on a real emulator.
    python tests/gameboy_play_env_test.py
"""
import numpy as np
from cusi.envs.gameboy import GameBoyPlayEnv, MSG_PARSE_FAILED

GAME = "pokemon_red"
MOVES = ["RIGHT", "RIGHT", "DOWN", "Thought: go up\nAction: UP", "A", "LEFT"]


def check(*, cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def common_checks(*, env: GameBoyPlayEnv) -> np.ndarray:
    obs, info = env.reset()
    frame0 = obs["frame"]
    check(cond=frame0.dtype == np.uint8 and frame0.ndim == 3 and frame0.shape[2] == 3,
          msg=f"frame is HxWx3 uint8 {frame0.shape}")
    check(cond=env.observation_space["frame"].shape == frame0.shape, msg="frame matches observation_space")
    check(cond=obs["texts"] == {} and len(obs["actions"]) > 0, msg=f"texts {{}}; actions: {obs['actions'][:70]!r}")

    obs, r, term, trunc, info = env.step("jump around")
    check(cond=not info["valid"] and info["error"] == MSG_PARSE_FAILED and np.array_equal(obs["frame"], frame0),
          msg="unparseable text -> invalid no-op, frame unchanged")
    for move in MOVES:
        obs, r, term, trunc, info = env.step(move)
    check(cond=info["valid"] and info["parsed_action"]["action_type"] == "LowLevelAction",
          msg=f"moves are valid; canonical {info['parsed_action']}")
    check(cond=r == 0.0 and not term, msg="no reward or termination from wandering")

    obs, info = env.reset()
    check(cond=np.array_equal(obs["frame"], frame0) and info["step"] == 0, msg="reset restores the exact initial frame")
    obs2 = None
    for move in MOVES:
        obs2, *_ = env.step(move)
    obs, _ = env.reset()
    for move in MOVES:
        obs, *_ = env.step(move)
    check(cond=np.array_equal(obs["frame"], obs2["frame"]), msg="same actions after reset -> identical frames (deterministic)")
    return frame0


def main():
    print("=== free_play (fixed start state, as the curiosity runs use)")
    env = GameBoyPlayEnv.from_benchmark(game=GAME, task_index=0, mode="free_play")
    common_checks(env=env)
    obs, _ = env.reset()
    check(cond=obs["goal"] == "", msg="no goal in free_play")
    env.close()

    print("=== free_play truncation at max_steps")
    env = GameBoyPlayEnv(game=GAME, init_state=env.init_state, mode="free_play", max_steps=4)
    env.reset()
    truncated, n = False, 0
    while not truncated and n < 20:
        _, _, _, truncated, _ = env.step("DOWN")
        n += 1
    check(cond=truncated, msg=f"truncated after {n} steps with max_steps=4")
    env.close()

    print("=== test (benchmark task 0)")
    env = GameBoyPlayEnv.from_benchmark(game=GAME, task_index=0, mode="test")
    common_checks(env=env)
    obs, _ = env.reset()
    check(cond=obs["goal"] == "Enter the Pokemon Center", msg=f"goal shown: {obs['goal']!r}")
    env.close()
    print("\nALL OK")


if __name__ == "__main__":
    main()
