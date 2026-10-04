"""
Check save_state / load_state / delete_state on a real env, and time them. From the CUSI root:

    source scripts/utils.sh && python tests/env_saved_state_test.py --env gameboy
    bash scripts/container.sh python tests/env_saved_state_test.py --env web
    sbatch --output="$results_dir/logs/env_saved_state_%j.out" slurm/env_saved_state.sh   # android (needs /dev/kvm)

Steps: reset, take random actions, save state A; take more actions (recorded); load A and
compare the observation with the one at save time; replay the recorded actions and compare
the frames with the first pass; then delete A and check that loading it errors.

GameBoy must match exactly. Android should match up to small screen differences (status bar
clock); the web restores only the browser's side of a live page (see webvoyager.py), so its
numbers are reported, not asserted.
"""
import glob
import os
import subprocess
import time
import click
import numpy as np


def check(*, cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def frame_diff(*, a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that differ (1.0 if the shapes differ)."""
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(np.any(a != b, axis=-1)))


def make_env(*, env_name: str):
    if env_name == "gameboy":
        from cusi_envs.gameboy import GameBoyPlayEnv
        return GameBoyPlayEnv(game="pokemon_red", mode="free_play")
    if env_name == "android":
        from cusi_envs.android_world import AndroidPlayEnv
        return AndroidPlayEnv(task="ContactsAddContact", mode="free_play", start_app="Contacts")
    from cusi_envs.webvoyager import WebVoyagerPlayEnv
    return WebVoyagerPlayEnv(url="https://arxiv.org/", mode="free_play", fast_waits=True)


def android_snapshot_sizes() -> str:
    paths = glob.glob(f"/tmp/{os.environ.get('USER', 'user')}-cusi-emulator-*/avd/*.avd/snapshots/*")
    if not paths:
        return "no snapshot dirs found"
    return subprocess.run(["du", "-sh", *sorted(paths)], capture_output=True, text=True).stdout.strip()


@click.command()
@click.option("--env", "env_name", type=click.Choice(["gameboy", "android", "web"]), required=True)
@click.option("--n_before", default=5, help="Random actions before saving.")
@click.option("--n_after", default=5, help="Random actions after saving (then replayed after the load).")
def main(env_name, n_before, n_after):
    env = make_env(env_name=env_name)
    env.reset(seed=0)
    for _ in range(n_before):
        env.step(env.sample_action())
    saved = env.current_obs

    t = time.time()
    state_a = env.save_state()
    t_save = time.time() - t
    check(cond=state_a in env.saved_state_ids, msg=f"save_state -> id {state_a} ({t_save:.2f}s)")
    if env_name == "android":
        print("        snapshot sizes:\n" + android_snapshot_sizes())

    actions = [env.sample_action() for _ in range(n_after)]
    if env_name == "android":
        # Random taps often change nothing; end on Home -> Settings so the screen surely moves.
        actions[-2:] = ['{"action_type": "navigate_home"}', '{"action_type": "open_app", "app_name": "Settings"}']
    frames = []
    previous = saved["frame"]
    for action in actions:
        _obs, _r, _term, _trunc, info = env.step(action)
        frames.append(env.current_obs["frame"])
        print(f"        step {action.splitlines()[-1][:70]!r}: valid {info['valid']}, error {info['error']!r}, "
              f"{frame_diff(a=frames[-1], b=previous):.1%} changed")
        previous = frames[-1]
    moved = frame_diff(a=frames[-1], b=saved["frame"])
    check(cond=moved > 0.05, msg=f"after {n_after} more actions the screen moved ({moved:.1%} of pixels differ)")

    t = time.time()
    result = env.load_state(state_id=state_a)
    t_load = time.time() - t
    check(cond=result is None, msg=f"load_state returns nothing ({t_load:.2f}s)")
    restored = env.current_obs
    diff = frame_diff(a=restored["frame"], b=saved["frame"])
    same_text = restored["texts"] == saved["texts"]
    print(f"        restored frame: {diff:.2%} of pixels differ from the saved frame; texts equal: {same_text}")
    if env_name == "gameboy":
        check(cond=diff == 0.0 and same_text, msg="restored observation is identical")
    elif env_name == "android":
        check(cond=diff < 0.02, msg="restored frame within 2% of the saved one")

    replay = []
    for action in actions:
        env.step(action)
        replay.append(env.current_obs["frame"])
    replay_diffs = [frame_diff(a=x, b=y) for x, y in zip(frames, replay)]
    print("        replay vs first pass, per step: " + ", ".join(f"{d:.1%}" for d in replay_diffs))
    if env_name == "gameboy":
        check(cond=max(replay_diffs) == 0.0, msg="replaying the same actions after the load is deterministic")

    state_b = env.save_state()
    env.delete_state(state_id=state_a)
    check(cond=state_a not in env.saved_state_ids and state_b in env.saved_state_ids,
          msg="delete_state removes only that id")
    try:
        env.load_state(state_id=state_a)
        failed = False
    except RuntimeError:
        failed = True
    check(cond=failed, msg="loading a deleted state errors")
    env.load_state(state_id=state_b)
    check(cond=env.current_obs is not None, msg="the other state still loads")
    env.close()
    check(cond=env.saved_state_ids == (), msg="close() deletes all saved states")
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
