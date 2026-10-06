"""
End-to-end check of cusi.envs.android_world.AndroidPlayEnv on a real emulator.

Run inside the container on a KVM node, with an emulator already started with
snapshots allowed (setup/android/play_env_test.sbatch does all of this):

    bash scripts/android_emulator.sh --action start --read_only true --snapshots true
    python tests/android_play_env_e2e.py

Checks, for test and free_play modes and both reset modes: observation contents,
invalid / out-of-range / valid actions, `status` handling per mode, and that reset()
returns to the initial state (snapshot mode: including changes outside the task's apps,
and the clock).
"""
import collections
import re
import subprocess
import time
from cusi.envs.android_world import AndroidPlayEnv, MSG_PARSE_FAILED, MSG_OUT_OF_RANGE, MSG_NO_STATUS

# Status-bar items change on their own (the emulator cycles its simulated signal
# strength; notifications linger), in evaluation too, so they are not part of the scene.
STATUS_BAR = re.compile(r"signal|bars|[Bb]attery|Wifi|Wi-Fi|notification|Android System|"
                        r"\"text\": \"\d{1,2}:\d{2}\"|No SIM|Airplane|Do Not Disturb", re.IGNORECASE)


def element_diff(*, before: str, after: str) -> tuple[list[str], list[str]]:
    """Elements only in `before` / only in `after`, ignoring index numbers (one
    extra element shifts every later index)."""
    def strip(text):
        return collections.Counter(re.sub(r'^UI element \d+: \{"index": \d+, ', "{", line)
                                   for line in text.splitlines() if line.strip())
    b, a = strip(before), strip(after)
    return list((b - a).elements()), list((a - b).elements())

TASK = "ContactsAddContact"
ADB = ["adb", "-s", "emulator-5554"]


def sh(*, cmd: str) -> str:
    return subprocess.run(ADB + ["shell", cmd], capture_output=True, text=True).stdout.strip()


def check(*, cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def timed(*, fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def run_mode(*, mode: str, reset_mode: str) -> None:
    print(f"\n=== mode={mode} reset_mode={reset_mode}", flush=True)
    env, t_init = timed(fn=lambda: AndroidPlayEnv(task=TASK, mode=mode, reset_mode=reset_mode))
    print(f"  init (build scene{' + snapshot' if reset_mode == 'snapshot' else ''}): {t_init:.1f}s")
    (obs, info), t_reset = timed(fn=env.reset)
    print(f"  first reset: {t_reset:.1f}s")
    check(cond=obs["frame"].ndim == 3 and obs["frame"].shape[2] == 3 and obs["frame"].dtype.name == "uint8",
          msg=f"frame is HxWx3 uint8 {obs['frame'].shape}")
    check(cond=env.observation_space["frame"].shape == obs["frame"].shape, msg="frame matches observation_space")
    check(cond=set(obs["texts"]) == {"ui_elements"} and "UI element" in obs["texts"]["ui_elements"], msg=f"texts['ui_elements'] is M3A's element list ({len(obs['texts']['ui_elements'])} chars)")
    if mode == "test":
        check(cond="Hugo" in obs["goal"] or len(obs["goal"]) > 10, msg=f"goal shown: {obs['goal']!r}")
        check(cond='"status"' in obs["actions"], msg="menu offers status")
        check(cond=env.max_steps is not None and env.max_steps > 0, msg=f"eval step budget: {env.max_steps}")
    else:
        check(cond=obs["goal"] == "", msg="no goal in free_play")
        check(cond='"status"' not in obs["actions"], msg="menu has no status")
        check(cond=env.max_steps is None, msg="no step limit in free_play")
    initial_text = obs["texts"]["ui_elements"]
    clock0 = sh(cmd="date -u +%Y-%m-%dT%H:%M")

    _, r, term, trunc, info = env.step("I would like to click something")
    check(cond=not info["valid"] and info["error"] == MSG_PARSE_FAILED and not term, msg="garbage -> invalid, M3A parse message")
    _, r, term, trunc, info = env.step('{"action_type": "click", "index": 999}')
    check(cond=not info["valid"] and info["error"] == MSG_OUT_OF_RANGE, msg="index 999 -> out of range")
    obs, r, term, trunc, info = env.step('Reason: x\nAction: {"action_type": "open_app", "app_name": "Contacts"}')
    check(cond=info["valid"] and info["parsed_action"] == {"action_type": "open_app", "app_name": "Contacts"}
          and info["step"] == 3, msg=f"open_app Contacts -> valid, canonical {info['parsed_action']}")
    check(cond=obs["texts"]["ui_elements"] != initial_text, msg="screen changed after opening Contacts")

    # A change outside the task's apps: a global setting and a file.
    sh(cmd="settings put global cusi_probe changed")
    sh(cmd="echo changed > /sdcard/cusi_probe.txt")

    status = '{"action_type": "status", "goal_status": "complete"}'
    _, r, term, trunc, info = env.step(status)
    if mode == "test":
        check(cond=term and info["valid"] and info.get("goal_status") == "complete",
              msg=f"status -> terminated, reward {r} (contact not created, so 0.0 expected)")
        check(cond=r == 0.0, msg="reward 0.0 for an unsolved task")
    else:
        check(cond=not term and not info["valid"] and info["error"] == MSG_NO_STATUS, msg="status rejected in free_play")

    (obs, info), t_reset = timed(fn=env.reset)
    print(f"  reset after steps: {t_reset:.1f}s")
    removed, added = element_diff(before=initial_text, after=obs["texts"]["ui_elements"])
    for line in removed:
        print(f"        - {line}")
    for line in added:
        print(f"        + {line}")
    scene_diff = [line for line in removed + added if not STATUS_BAR.search(line)]
    check(cond=not scene_diff,
          msg=f"reset: same UI elements as the initial state ({len(removed) + len(added)} status-bar differences ignored)"
          if not scene_diff else f"reset: {len(scene_diff)} scene elements differ from the initial state")
    check(cond=info["step"] == 0, msg="reset: step counter 0")
    clock1 = sh(cmd="date -u +%Y-%m-%dT%H:%M")
    check(cond=clock1 == clock0, msg=f"reset: clock back to {clock0} (now {clock1})")
    probe = sh(cmd="settings get global cusi_probe")
    probe_file = sh(cmd="cat /sdcard/cusi_probe.txt 2>/dev/null")
    if reset_mode == "snapshot":
        check(cond=probe != "changed" and probe_file != "changed",
              msg=f"snapshot reset undid the out-of-task changes (setting={probe!r}, file={probe_file!r})")
    else:
        print(f"  info  reinit keeps out-of-task changes, as evaluation does (setting={probe!r})")
        sh(cmd="settings delete global cusi_probe"); sh(cmd="rm -f /sdcard/cusi_probe.txt")
    env.close()


if __name__ == "__main__":
    run_mode(mode="test", reset_mode="snapshot")
    run_mode(mode="free_play", reset_mode="snapshot")
    run_mode(mode="test", reset_mode="reinit")
    print("\nALL OK")
