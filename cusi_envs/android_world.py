"""AndroidWorld as a TextActionEnv (see cusi_envs/base.py).

    env = AndroidPlayEnv(task="ContactsAddContact", mode="test")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(
        'Reason: open contacts\nAction: {"action_type": "open_app", "app_name": "Contacts"}')

The scene (task type + its seeded parameters) is fixed at construction and built
with AndroidWorld's own evaluation code: the same per-instance seed derivation as
create_suite, task.initialize_task (app snapshot restore, pinned clock, seeded
task data), then Home and hide_automation_ui exactly as M3A.reset does.

reset() returns to that initial state:
    "snapshot" (default)  load an emulator snapshot saved right after the scene was
                          built: disk + RAM, so every app, setting, notification and
                          the clock are restored. Needs an emulator started with
                          --snapshots true (scripts/android_emulator.sh).
    "reinit"              what evaluation does between tasks: task.tear_down, then
                          initialize_task again with the same parameters. Restores
                          only the task's apps, its data and the clock.

Observations, parsing, index checks and feedback strings follow M3A (the agent the
benchmark was evaluated with). Modes (base.py): "test" ends the episode on the
`status` action with reward = task.is_successful (as suite_utils scores it: only if
the agent declared done), and truncates at the evaluation's step budget; "free_play"
hides the goal, rejects `status`, and gives reward 0.

Observation: frame (set-of-mark screenshot by default), texts={"ui_elements": M3A's
numbered element list}, actions (M3A's action menu), goal.

The emulator runs inside the CUSI container: run this code via scripts/container.sh,
with the emulator started by scripts/android_emulator.sh (with --snapshots true for
reset_mode="snapshot"), or pass an AndroidEmulator so the env can restart a hung
emulator itself.

Parts of this module (parse/validate/feedback handling, observation building, the
watchdog pattern, the snapshot commands) are adapted from SetupAttempt's
playenv/android_play.py.
"""
import datetime
import json
import re
import hashlib
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Optional
import numpy as np
from android_world import registry, suite_utils
from android_world.agents import agent_utils, m3a, m3a_utils
from android_world.env import adb_utils, device_constants, env_launcher, interface, json_action
from android_world.utils import datetime_utils
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_info, log_warn, log_error
from cusi_envs.base import TextActionEnv, scene_hash

# android_world/run.py's --task_random_seed default: the evaluation's suite seed.
DEFAULT_SUITE_SEED = 30

# M3A's feedback strings, verbatim, so a policy sees what it would see in evaluation.
MSG_PARSE_FAILED = ("Can not parse the output to a valid action. Please make sure to pick"
                    " the action from the list with required parameters (if any) in the"
                    " correct JSON format!")
MSG_OUT_OF_RANGE = ("The parameter index is out of range. Remember the index must be in"
                    " the UI element list!")
MSG_EXEC_FAILED = ("Can not execute the action, make sure to select the action with"
                   " the required parameters (if any) in the correct JSON format!")
MSG_NO_STATUS = ("The `status` action is not available: there is no task to finish."
                 " No action was performed.")

# obs["texts"] channels: M3A's numbered list of the UI elements on screen.
TEXT_KEYS = ("ui_elements",)

_INDEXED_ACTIONS = ("click", "long_press", "input_text", "scroll")
_JSON_ACTION_FIELDS = ("action_type", "index", "x", "y", "text", "direction", "goal_status", "app_name")


def instance_seed(*, suite_seed: int, task_name: str, instance: int) -> int:
    """create_suite's per-instance seed (suite_utils.create_suite._get_instance_seed)."""
    unique = f"{suite_seed}_{task_name}_{instance}"
    return int(hashlib.sha256(unique.encode()).hexdigest(), 16) % (2 ** 32)


def action_menu(*, mode: str) -> str:
    """M3A's action list (from its prompt), one action per line. free_play drops the
    finishing `status` actions and `answer` (there is no question to answer)."""
    prefix = m3a.PROMPT_PREFIX.replace("{{", "{").replace("}}", "}")
    start = prefix.index("- If you think the task has been completed")
    lines = []
    for line in prefix[start:].split("\n- "):
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        if mode == "free_play" and ('"status"' in line or '"answer"' in line):
            continue
        lines.append("- " + line)
    return "\n".join(lines)


def parse_action(*, text: str) -> tuple[Optional[dict], Optional[str]]:
    """Parse model output into an action dict. Accepts M3A's 'Reason: ...\\nAction: {...}'
    and, more leniently than M3A, a bare JSON action. Returns (action, error)."""
    _reason, action = m3a_utils.parse_reason_action_output(text)
    candidate = action if action else text
    parsed = agent_utils.extract_json(candidate) if candidate else None
    if not isinstance(parsed, dict) or "action_type" not in parsed:
        return None, MSG_PARSE_FAILED
    return parsed, None


def canonical_action(*, action: json_action.JSONAction) -> dict:
    """The JSONAction's set fields, as a plain dict (stable input for world models)."""
    return {k: getattr(action, k) for k in _JSON_ACTION_FIELDS if getattr(action, k, None) is not None}


class EnvStalled(RuntimeError):
    """A reset/step did not return within call_timeout (the emulator hung)."""


class AndroidEmulator:
    """Handle on one headless emulator managed by scripts/android_emulator.sh, so an
    AndroidPlayEnv can restart it when it hangs. Must run inside the container."""

    def __init__(self, *, port: int = 5554, grpc_port: int = 8554, avd: str = "AndroidWorldAvd",
                 read_only: bool = True, snapshots: bool = True, parameters: dict[str, Any] = None) -> None:
        self._parameters = load_parameters(parameters)
        self.port = port
        self.grpc_port = grpc_port
        self.avd = avd
        self.read_only = read_only
        self.snapshots = snapshots
        self._script = os.path.join(self._parameters["project_root"], "scripts", "android_emulator.sh")
        self._pid_file = f"/tmp/{os.environ.get('USER', 'user')}-cusi-emulator-{port}/emulator.pid"

    def _run(self, *, action: str, timeout: float) -> None:
        cmd = ["bash", self._script, "--action", action, "--port", str(self.port),
               "--grpc_port", str(self.grpc_port), "--avd", self.avd,
               "--read_only", str(self.read_only).lower(), "--snapshots", str(self.snapshots).lower()]
        subprocess.run(cmd, cwd=self._parameters["project_root"], check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL)

    def start(self) -> None:
        self._run(action="start", timeout=900)

    def stop(self) -> None:
        self._run(action="stop", timeout=300)

    def kill_hard(self) -> None:
        """SIGKILL the emulator: a hung emulator also hangs `adb emu kill`."""
        if not os.path.exists(self._pid_file):
            return
        with open(self._pid_file) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for _ in range(60):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        os.remove(self._pid_file)

    def restart(self) -> None:
        self.kill_hard()
        self.start()


class AndroidPlayEnv(TextActionEnv):
    """AndroidWorld task scene as a text-action gym.Env. See the module docstring.

    info["raw_frame"] is the screenshot without set-of-mark labels (M3A shows the model
    both)."""

    env_description = "an Android phone"

    def __init__(
        self,
        *,
        task: str,
        mode: str = "test",
        suite_seed: int = DEFAULT_SUITE_SEED,
        instance: int = 0,
        console_port: int = 5554,
        grpc_port: int = 8554,
        adb_path: Optional[str] = None,
        emulator: Optional[AndroidEmulator] = None,
        reset_mode: str = "snapshot",
        som: bool = True,
        max_steps: Optional[int] = None,
        wait_after_action_seconds: float = 2.0,
        call_timeout: Optional[float] = 600.0,
        probe_success: bool = False,
        start_app: Optional[str] = None,
        connection: Optional[interface.AsyncEnv] = None,
        stabilize_after_action: bool = True,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        :param task: AndroidWorld task name, e.g. "ContactsAddContact".
        :param mode: "test" or "free_play" (see cusi_envs/base.py).
        :param suite_seed: The evaluation's suite seed (run.py --task_random_seed).
        :param instance: Which instance of the task (run.py --n_task_combinations index).
        :param emulator: If given, the env restarts this emulator when a call hangs and
            rebuilds the scene; otherwise a hang raises EnvStalled.
        :param reset_mode: "snapshot" (full restore; needs --snapshots true) or "reinit".
        :param som: Draw M3A's set-of-mark labels on the frame (as M3A shows the model).
        :param max_steps: Truncate after this many steps. None: the evaluation's budget
            (10 * task complexity) in test mode, unlimited in free_play.
        :param probe_success: free_play only: report task.is_successful as
            info["probe_success"] (for logging; never show it to the policy).
        :param start_app: Open this app (M3A's open_app action) as the last step of building
            the scene, so the initial state is that app's screen rather than the home
            screen most tasks start on. Part of the scene: reset() returns to it.
        :param connection: An open AndroidWorld connection (load_and_setup_env) to reuse, as
            the evaluation harness keeps one for the whole suite; default: open a new one.
            close(close_connection=False) leaves it open for the next env.
        :param stabilize_after_action: Wait for the screen to stabilize before the
            observation after an action. False matches M3A, which looks right after its
            2 s pause (for the step summary) and observes again, stabilized, at the start of
            the next step (call observe() for that; cusi_eval does).
        """
        self._parameters = load_parameters(parameters)
        if reset_mode not in ("snapshot", "reinit"):
            log_error(f"reset_mode must be 'snapshot' or 'reinit', got {reset_mode!r}", parameters=self._parameters)
        aw_registry = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
        if task not in aw_registry:
            log_error(f"Unknown AndroidWorld task {task!r}", parameters=self._parameters)
        self.task_name = task
        self._task_type = aw_registry[task]
        self._seed = instance_seed(suite_seed=suite_seed, task_name=task, instance=instance)
        self.scene_id = scene_hash(parts=("androidworld", task, suite_seed, instance))
        self._conn = dict(console_port=console_port, grpc_port=grpc_port,
                          adb_path=adb_path or shutil.which("adb"))
        self._serial = f"emulator-{console_port}"
        self._emulator = emulator
        self.reset_mode = reset_mode
        self.som = som
        self.wait_after_action_seconds = wait_after_action_seconds
        self.call_timeout = call_timeout
        self.probe_success = probe_success
        self.start_app = start_app
        self.stabilize_after_action = stabilize_after_action
        if start_app:
            self.scene_id = scene_hash(parts=("androidworld", task, suite_seed, instance, start_app))
        self._snapshot_name = f"cusi_{self.scene_id}"
        self._states: dict[str, datetime.datetime] = {}   # saved state_id -> device clock at save
        self.recoveries = 0

        self._env = connection if connection is not None else self._guarded(fn=self._connect)
        env = self._env
        self._task, self._snapshot_clock, raw = self._guarded(fn=lambda: self._build_scene(env))
        super().__init__(mode=mode, frame_shape=raw[0].shape, text_keys=TEXT_KEYS,
                         parameters=self._parameters)
        if max_steps is None and mode == "test":
            max_steps = int(10 * self._task.complexity)   # suite_utils._allocate_step_budget
        self.max_steps = max_steps
        self.actions_text = action_menu(mode=mode)
        self._at_initial_state = True
        self._steps = 0
        self._ui_count = raw[1][1]
        log_info(f"AndroidPlayEnv ready: {task} (seed {self._seed}, mode {mode}, reset {reset_mode})",
                 parameters=self._parameters)

    # ------------------------------------------------------------------ scene

    def _connect(self) -> interface.AsyncEnv:
        # run.py's call, minus emulator_setup (done once, by setup/android/app_setup.sh).
        return env_launcher.load_and_setup_env(emulator_setup=False, **self._conn)

    def _adb(self, *args: str, timeout: float = 120) -> str:
        out = subprocess.run([self._conn["adb_path"], "-s", self._serial, *args],
                             capture_output=True, text=True, timeout=timeout)
        return (out.stdout + out.stderr).strip()

    # The methods below run on watchdog worker threads. They take the connection (and
    # task) as arguments and only return values, so a worker abandoned after a stall
    # can never touch the connection or state that replaced it.

    def _build_scene(self, env: interface.AsyncEnv):
        """Build the scene from scratch; save the snapshot.
        Returns (task, snapshot_clock, raw observation)."""
        task = suite_utils._instantiate_task(self._task_type, seed=self._seed, env=env)
        self._check_app_snapshots(task)
        task.initialize_task(env)
        self._go_to_start(env, task)
        snapshot_clock = None
        if self.reset_mode == "snapshot":
            snapshot_clock = self._save_snapshot(self._snapshot_name)
        return task, snapshot_clock, self._observe_raw(env)

    def _go_to_start(self, env: interface.AsyncEnv, task) -> None:
        # M3A.reset: Home if the task says so, clear the interaction cache, hide the
        # pointer-location overlay.
        env.reset(go_home=task.start_on_home_screen)
        env.hide_automation_ui()
        if self.start_app:
            env.execute_action(json_action.JSONAction(action_type="open_app", app_name=self.start_app))
            time.sleep(3.0)
            env.interaction_cache = ""

    def _check_app_snapshots(self, task) -> None:
        """initialize_task resets each task app's data from a snapshot taken at app setup,
        and only warns (in AndroidWorld's logs) when one is missing. Say so loudly."""
        self._adb("root")
        for app_name in task.app_names:
            if app_name == "clipper":   # task_eval._initialize_apps skips it too
                continue
            try:
                package = adb_utils.extract_package_name(adb_utils.get_adb_activity(app_name))
            except Exception:
                continue
            path = f"{device_constants.SNAPSHOT_DATA}/{package}"
            if "missing" in self._adb("shell", f"ls -d {path} 2>/dev/null || echo missing"):
                log_warn(f"No app-data snapshot for {app_name} at {path}: this task's {app_name} data will"
                         " NOT be reset. Re-run app setup (setup/setup.sh rebuilds the AVD).",
                         parameters=self._parameters)

    def _focused_package(self) -> Optional[str]:
        for line in self._adb("shell", "dumpsys", "window").splitlines():
            match = re.search(r"mCurrentFocus=Window\{\S+ \S+ ([^/\s}]+)", line)
            if match:
                return match.group(1)
        return None

    def _wait_until_restored(self, env: interface.AsyncEnv, timeout: float = 120.0) -> None:
        """After a snapshot load the device is briefly unreachable, and AndroidWorld's
        accessibility tree keeps reporting the pre-load screen until it catches up.
        Wait until the device has booted with a focused window and the tree shows that
        window's app, so the next observation is of the restored screen."""
        deadline = time.time() + timeout
        focused = None
        while time.time() < deadline:
            if self._adb("shell", "getprop", "sys.boot_completed").strip() == "1":
                focused = self._focused_package()
                if focused:
                    break
            time.sleep(1)
        if not focused:
            raise RuntimeError("Device did not come back after the snapshot load")
        while time.time() < deadline:
            packages = [u.package_name for u in env.get_state(wait_to_stabilize=False).ui_elements
                        if u.package_name and u.package_name != "com.android.systemui"]
            if packages and max(set(packages), key=packages.count) == focused:
                return
            time.sleep(1)
        raise RuntimeError(f"UI tree did not catch up with the restored screen ({focused}) after the snapshot load")

    def _device_time(self) -> datetime.datetime:
        epoch = int(self._adb("shell", "date", "+%s").split()[-1])
        return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)

    def _save_snapshot(self, name: str) -> datetime.datetime:
        """Save an emulator snapshot (disk + RAM); returns the device clock at save time."""
        out = self._adb("emu", "avd", "snapshot", "save", name, timeout=600)
        if "OK" not in out:
            raise RuntimeError(f"Snapshot save failed ({out!r}). Start the emulator with --snapshots true.")
        return self._device_time()

    def _load_snapshot(self, env: interface.AsyncEnv, name: str, clock: datetime.datetime) -> None:
        out = self._adb("emu", "avd", "snapshot", "load", name, timeout=600)
        if "OK" not in out:
            raise RuntimeError(f"Snapshot load failed: {out!r}")
        self._wait_until_restored(env)
        # Pin the clock back to its value at save time, whatever the guest clock
        # does across a load.
        if abs((self._device_time() - clock).total_seconds()) > 2:
            datetime_utils.set_datetime(env.controller, clock)
        # Python-side state the snapshot cannot restore.
        env.interaction_cache = ""

    def _restore(self, env: interface.AsyncEnv, task, snapshot_clock):
        """Return the device to the scene's initial state. Returns the raw observation."""
        if self.reset_mode == "snapshot":
            self._load_snapshot(env, self._snapshot_name, snapshot_clock)
        else:
            task.tear_down(env)
            task.initialize_task(env)
            self._go_to_start(env, task)
        return self._observe_raw(env)

    # ------------------------------------------------------------ observation

    def _observe_raw(self, env: interface.AsyncEnv, stabilize: bool = True):
        """(frame, (ui_elements_text, ui_count), raw_pixels): M3A's view of the current
        screen; raw_pixels is the screenshot without set-of-mark labels."""
        state = env.get_state(wait_to_stabilize=stabilize)
        size = env.logical_screen_size
        raw_pixels = state.pixels.copy().astype(np.uint8)
        frame = state.pixels.copy()
        if self.som:
            for i, ui in enumerate(state.ui_elements):
                if m3a_utils.validate_ui_element(ui, size):
                    m3a_utils.add_ui_element_mark(frame, ui, i, size, env.physical_frame_boundary,
                                                  env.orientation)
        text = m3a._generate_ui_elements_description_list(state.ui_elements, size)
        return frame.astype(np.uint8), (text, len(state.ui_elements)), raw_pixels

    def _obs(self, *, raw) -> dict:
        frame, (ui_elements, _count), _raw_pixels = raw
        goal = self._task.goal if self.mode == "test" else ""
        return {"frame": frame, "texts": {"ui_elements": ui_elements},
                "actions": self.actions_text, "goal": goal}

    def observe(self) -> tuple:
        """Observe the current screen again (stabilized) without acting, as M3A does at the
        start of every step. Updates current_obs and the element count used by index checks.
        Returns (obs, info)."""
        env = self._env
        raw = self._guarded(fn=lambda: self._observe_raw(env))
        self._ui_count = raw[1][1]
        obs = self._obs(raw=raw)
        self.current_obs = obs
        return obs, self.make_info(valid=True, error=None, parsed_action=None, step=self._steps,
                                   scene_id=self.scene_id, raw_frame=raw[2])

    # ---------------------------------------------------------------- gym API

    def _reset_impl(self):
        env, task, clock = self._env, self._task, self._snapshot_clock
        if self._at_initial_state:
            raw = self._guarded(fn=lambda: self._observe_raw(env))
        else:
            try:
                raw = self._guarded(fn=lambda: self._restore(env, task, clock))
            except EnvStalled as e:
                raw = self._recover(why=f"reset: {e}")
            except RuntimeError as e:
                # e.g. "Device did not come back after the snapshot load": the restore itself
                # failed, so the device is in an unknown state. Same remedy as a stall.
                if self._emulator is None:
                    raise
                raw = self._recover(why=f"reset failed: {e}")
        self._at_initial_state = False
        self._steps = 0
        self._ui_count = raw[1][1]
        info = self.make_info(valid=True, error=None, parsed_action=None, step=0, scene_id=self.scene_id,
                              raw_frame=raw[2])
        return self._obs(raw=raw), info

    def _step_impl(self, action: str):
        self._at_initial_state = False
        parsed, error = parse_action(text=action)
        ja = None
        if parsed is not None:
            try:
                ja = json_action.JSONAction(**parsed)
            except Exception:
                ja, error = None, MSG_PARSE_FAILED
        if ja is not None and ja.action_type in _INDEXED_ACTIONS and ja.index is not None:
            # As M3A: only too-large indices are rejected (a negative one indexes from the end).
            if int(ja.index) >= self._ui_count:
                ja, error = None, MSG_OUT_OF_RANGE
        if ja is not None and ja.action_type == json_action.STATUS and self.mode == "free_play":
            ja, error = None, MSG_NO_STATUS

        terminated, reward, extra = False, 0.0, {}
        env, task = self._env, self._task
        try:
            if ja is not None and ja.action_type == json_action.STATUS:
                # test mode: the agent declares the episode over; score it the way
                # suite_utils does (success counts only because the agent said done).
                terminated = True
                reward = float(self._guarded(fn=lambda: task.is_successful(env)))
                extra["goal_status"] = ja.goal_status
                raw = self._guarded(fn=lambda: self._observe_raw(env))
            else:
                if ja is not None:
                    error = self._guarded(fn=lambda: self._execute(env, ja))
                raw = self._guarded(fn=lambda: self._observe_raw(env, stabilize=self.stabilize_after_action))
        except EnvStalled as e:
            # The device was restarted and the scene rebuilt, so this episode cannot
            # continue: truncate it.
            raw = self._recover(why=f"step: {e}")
            self._steps += 1
            self._ui_count = raw[1][1]
            info = self.make_info(valid=False, error="The environment stopped responding and was restarted.",
                                  parsed_action=canonical_action(action=ja) if ja else None,
                                  step=self._steps, scene_id=self.scene_id, env_recovered=str(e),
                                  raw_frame=raw[2])
            return self._obs(raw=raw), 0.0, False, True, info

        self._steps += 1
        self._ui_count = raw[1][1]
        truncated = (not terminated) and self.max_steps is not None and self._steps >= self.max_steps
        if self.mode == "free_play" and self.probe_success:
            extra["probe_success"] = float(self._guarded(fn=lambda: task.is_successful(env)))
        valid = ja is not None and error is None
        info = self.make_info(valid=valid, error=error,
                              parsed_action=canonical_action(action=ja) if ja is not None else None,
                              step=self._steps, scene_id=self.scene_id, raw_frame=raw[2], **extra)
        return self._obs(raw=raw), reward, terminated, truncated, info

    def sample_action(self) -> str:
        """A random M3A action: tap or scroll on a listed element, a whole-screen scroll,
        back or home (weights 4:1:2:1:1). Never open_app/input_text/status."""
        rng = self.np_random
        kind = rng.choice(["click", "scroll_element", "scroll", "navigate_back", "navigate_home"],
                          p=[4 / 9, 1 / 9, 2 / 9, 1 / 9, 1 / 9])
        direction = str(rng.choice(["up", "down", "left", "right"]))
        if kind in ("click", "scroll_element") and self._ui_count > 0:
            index = int(rng.integers(self._ui_count))
            if kind == "click":
                return json.dumps({"action_type": "click", "index": index})
            return json.dumps({"action_type": "scroll", "direction": direction, "index": index})
        if kind in ("click", "scroll_element", "scroll"):
            return json.dumps({"action_type": "scroll", "direction": direction})
        return json.dumps({"action_type": str(kind)})

    def _execute(self, env: interface.AsyncEnv, ja: json_action.JSONAction) -> Optional[str]:
        try:
            env.execute_action(ja)
        except Exception as e:
            log_info(f"execute_action failed: {e}", parameters=self._parameters)
            return MSG_EXEC_FAILED
        time.sleep(self.wait_after_action_seconds)
        return None

    # ------------------------------------------------------------ saved states
    # Emulator snapshots (disk + RAM), as for reset_mode="snapshot"; they need an emulator
    # started with --snapshots true, and live in that emulator's private AVD copy.

    def _state_snapshot_name(self, state_id: str) -> str:
        return f"{self._snapshot_name}_{state_id}"

    def _save_state_impl(self, *, state_id: str) -> None:
        name = self._state_snapshot_name(state_id)
        try:
            self._states[state_id] = self._guarded(fn=lambda: self._save_snapshot(name))
        except (EnvStalled, RuntimeError) as e:
            log_error(f"Could not save Android state {state_id}: {e}", parameters=self._parameters)

    def _load_state_impl(self, *, state_id: str) -> dict:
        env, name, clock = self._env, self._state_snapshot_name(state_id), self._states[state_id]

        def load():
            self._load_snapshot(env, name, clock)
            return self._observe_raw(env)

        try:
            raw = self._guarded(fn=load)
        except (EnvStalled, RuntimeError) as e:
            log_error(f"Could not load Android state {state_id}: {e}", parameters=self._parameters)
        self._at_initial_state = False
        self._steps = 0
        self._ui_count = raw[1][1]
        return self._obs(raw=raw)

    def _delete_state_impl(self, *, state_id: str) -> None:
        out = self._adb("emu", "avd", "snapshot", "delete", self._state_snapshot_name(state_id), timeout=120)
        if "OK" not in out:
            log_error(f"Could not delete Android state {state_id}: {out!r}", parameters=self._parameters)
        del self._states[state_id]

    def close(self, *, close_connection: bool = True) -> None:
        try:
            for state_id in list(self._states):
                self._adb("emu", "avd", "snapshot", "delete", self._state_snapshot_name(state_id), timeout=120)
            self._states.clear()
            self._saved_state_ids.clear()
            if self.reset_mode == "snapshot":
                self._adb("emu", "avd", "snapshot", "delete", self._snapshot_name, timeout=120)
            if close_connection:
                self._env.close()
        except Exception as e:
            log_warn(f"AndroidPlayEnv.close: {e}", parameters=self._parameters)

    # ----------------------------------------------------------------- watchdog

    def _guarded(self, *, fn: Callable[[], Any]) -> Any:
        """Run fn on a worker thread; raise EnvStalled after call_timeout.

        A signal-based timeout cannot work: android_env's gRPC calls have no deadline
        and block inside gRPC's C core. Workers only return values; all env state is
        committed by the caller, so an abandoned worker cannot corrupt it."""
        if not self.call_timeout:
            return fn()
        box: dict = {}

        def run():
            try:
                box["result"] = fn()
            except BaseException as e:   # re-raised on the caller's thread
                box["error"] = e

        worker = threading.Thread(target=run, daemon=True, name="android-play-env-call")
        worker.start()
        worker.join(self.call_timeout)
        if worker.is_alive():
            raise EnvStalled(f"call did not return within {self.call_timeout:.0f}s")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _recover(self, *, why: str):
        """Restart a hung emulator and rebuild the scene. Returns its observation."""
        if self._emulator is None:
            raise EnvStalled(why + " (pass emulator=AndroidEmulator(...) to recover automatically)")
        log_warn(f"AndroidPlayEnv stalled ({why}); restarting the emulator and rebuilding the scene",
                 parameters=self._parameters)
        self.recoveries += 1
        if self._states:
            # The restarted emulator runs on a freshly staged AVD copy: saved snapshots are gone.
            log_warn(f"Emulator restart drops {len(self._states)} saved state(s)", parameters=self._parameters)
            self._states.clear()
            self._saved_state_ids.clear()
        self._emulator.restart()
        self._env = self._guarded(fn=self._connect)
        env = self._env
        self._task, self._snapshot_clock, raw = self._guarded(fn=lambda: self._build_scene(env))
        return raw
