"""
Sessions: one live env each, built from a task id (cusi.eval.splits) or a dev free-play scene (policy.py), as the
evals and the explorer build them.

Success (task sessions, when the episode ends): GameBoy and Android, the env's reward (the task's own check);
Web, WebVoyager's auto_eval judge on the answer and the last JUDGE_IMGS screenshots (the server's --judge_model). A
truncated episode scores 0. Free play has no success.

What a client gets is the env's obs and info minus the true state (info["true"]) and AndroidWorld's probe_success,
plus "state" (the env's full_state(): GameBoy's memory map, Android's foreground app, Web's page) only when the
instance allows it and the session asked.
"""
import argparse
import importlib.util
import json
import os
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import redirect_stdout
from io import StringIO
from typing import Optional
from PIL import Image
from cusi.utils.log_handling import log_info, log_warn
from cusi.env_server.codec import encode
from cusi.env_server.policy import Policy, free_play_scenes, held_out_packages, held_out_scopes
from cusi.env_server.recorder import Recorder
from cusi.eval.splits import env_of, load_splits, resolve

#: Never sent to a client: the true state, AndroidWorld's success probe, GameBoyWorlds' subgoal tracker (grader output),
#: and which held-out app or website a blocked action would have reached.
HIDDEN_INFO = ("true", "probe_success", "subgoals", "blocked")
#: GameBoyRL's per-transition records: every intermediate frame and the RAM-derived state at each (memory signals,
#: the test tracker): state, so never sent.
DROPPED_INFO = ("transition_states",)
ANDROID_SUITE_SEED = 30          # cusi.eval.android --task_random_seed default


class SessionError(Exception):
    """A refused request; status is the HTTP status the server answers with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class AndroidBackend:
    """The one emulator an Android server drives, and its AndroidWorld connection (shared by its sessions)."""

    def __init__(self, *, console_port: int, grpc_port: int, parameters: dict) -> None:
        from android_world.env import env_launcher
        from cusi.envs.android_world import AndroidEmulator
        self.console_port, self.grpc_port = console_port, grpc_port
        self.emulator = AndroidEmulator(port=console_port, grpc_port=grpc_port, snapshots=False,
                                        parameters=parameters)
        self.connection = env_launcher.load_and_setup_env(console_port=console_port, grpc_port=grpc_port,
                                                          emulator_setup=False, adb_path=shutil.which("adb"))

    def close(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass


class Backend:
    """Builds the envs of one kind (gameboy, web or android) and holds what they share."""

    def __init__(self, *, env: str, parameters: dict, judge: Optional[dict] = None, console_port: int = 5554,
                 grpc_port: int = 8554) -> None:
        self.env = env
        self._parameters = parameters
        self.scenes = None          # free-play scenes, built on first use
        self.android = AndroidBackend(console_port=console_port, grpc_port=grpc_port,
                                      parameters=parameters) if env == "android" else None
        self._judge_args = judge
        self._judge_model = None

    def free_play_scenes(self) -> dict:
        if self.scenes is None:
            self.scenes = free_play_scenes(env=self.env, parameters=self._parameters)
        return self.scenes

    def build(self, *, task_id: Optional[str], scene: Optional[str], block_held_out: bool):
        """-> (env, goal text, mode). block_held_out (dev): steps into a held-out website or app are undone and invalid
        (the envs' blocked lists), so the session never shows one."""
        p = self._parameters
        web_block = {"blocked": held_out_scopes()} if block_held_out else {}
        aw_block = {"blocked_packages": sorted(held_out_packages())} if block_held_out else {}
        if task_id is not None:
            r = resolve(task_id=task_id, parameters=p)
            goal = load_splits(parameters=p)[task_id]["text"]
            if self.env == "gameboy":
                from cusi.envs.gameboy import GameBoyPlayEnv
                from cusi.eval.gameboy import CONTROLLER_VARIANT, MAX_STEPS
                return GameBoyPlayEnv.from_benchmark(game=r["game"], task_index=r["task_index"], mode="test",
                                                     controller_variant=CONTROLLER_VARIANT, max_steps=MAX_STEPS,
                                                     parameters=p), goal, "test"
            if self.env == "web":
                from cusi.envs.webvoyager import WebVoyagerPlayEnv
                from cusi.eval.web import MAX_ITER
                from cusi.utils.paths import web_source_task_file
                return WebVoyagerPlayEnv(task_id=r["row"]["id"], mode="test",
                                         data_file=web_source_task_file(parameters=p), max_steps=MAX_ITER,
                                         fast_waits=False, headless=True, fix_box_color=True, window_width=1024,
                                         window_height=768, parameters=p, **web_block), goal, "test"
            from cusi.envs.android_world import AndroidPlayEnv
            a = self.android
            env = AndroidPlayEnv(task=r["task_class"], mode="test", suite_seed=ANDROID_SUITE_SEED,
                                 instance=r["instance"], console_port=a.console_port, grpc_port=a.grpc_port,
                                 emulator=a.emulator, reset_mode="reinit", connection=a.connection,
                                 stabilize_after_action=False, parameters=p, **aw_block)
            return env, env._task.goal, "test"     # the seed-0 instance's goal: the split's text for this id
        kwargs = self.free_play_scenes()[scene]
        if self.env == "gameboy":
            from cusi.envs.gameboy import GameBoyPlayEnv
            return GameBoyPlayEnv(mode="free_play", parameters=p, **kwargs), "", "free_play"
        if self.env == "web":
            from cusi.envs.webvoyager import WebVoyagerPlayEnv
            return WebVoyagerPlayEnv(mode="free_play", fast_waits=True, parameters=p, **kwargs, **web_block), "", \
                "free_play"
        from cusi.envs.android_world import AndroidPlayEnv
        a = self.android
        return AndroidPlayEnv(mode="free_play", console_port=a.console_port, grpc_port=a.grpc_port,
                              emulator=a.emulator, reset_mode="reinit", connection=a.connection,
                              stabilize_after_action=False, parameters=p, **kwargs, **aw_block), "", "free_play"

    def release(self, *, env) -> None:
        """Close an env; Android keeps the connection (and takes over a replacement the env made)."""
        if self.env == "android":
            try:
                if env.mode == "test":
                    env._guarded(fn=lambda: env._task.tear_down(env._env))
            except Exception as e:
                log_warn(f"[env_server] android tear_down failed: {e}", parameters=self._parameters)
            self.android.connection = env._env
            env.close(close_connection=False)
        else:
            env.close()

    def judge_web(self, *, goal: str, answer: str, frames: list) -> Optional[float]:
        """WebVoyager's auto_eval on one episode: 1, 0, or None (no judge configured, or no verdict)."""
        if self._judge_args is None:
            return None
        root = self._parameters["project_root"]
        spec = importlib.util.spec_from_file_location("auto_eval",
                                                      os.path.join(root, "WebVoyager", "evaluation", "auto_eval.py"))
        auto_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auto_eval)
        if self._judge_model is None:
            from benchmark_adapters.webvoyager_adapter import build_model_from_args
            self._judge_model = build_model_from_args(args=argparse.Namespace(**self._judge_args),
                                                      parameters=self._parameters)
        from cusi.eval.web import JUDGE_IMGS
        tmp = tempfile.mkdtemp(prefix="cusi-judge-")
        try:
            with open(os.path.join(tmp, "interact_messages.json"), "w") as f:
                json.dump([{"role": "system", "content": ""},
                           {"role": "user", "content": f"Now given a task: {goal}  Please interact with the page."},
                           {"role": "assistant", "content": f"Action: ANSWER; {answer}"}], f)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(tmp, f"screenshot{i + 1}.png"))
            with redirect_stdout(StringIO()):
                verdict = auto_eval.auto_eval_by_gpt4v(tmp, self._judge_model, JUDGE_IMGS)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return None if verdict is None else float(verdict)

    def close(self) -> None:
        if self.android is not None:
            self.android.close()


class Session:
    def __init__(self, *, backend: Backend, policy: Policy, task_id: Optional[str], scene: Optional[str],
                 want_state: bool, record_dir: Optional[str], parameters: dict) -> None:
        self.id = uuid.uuid4().hex[:12]
        #: Grants this session's own endpoints only (the test harness hands it to the policy it runs).
        self.token = secrets.token_hex(16)
        self.backend, self.policy, self.task_id, self.scene = backend, policy, task_id, scene
        self.want_state = want_state
        self._parameters = parameters
        self.lock = threading.Lock()
        self.last_used = time.time()
        self.env, self.goal, self.mode = backend.build(task_id=task_id, scene=scene,
                                                       block_held_out=policy.kind == "dev")
        self.max_steps = getattr(self.env, "max_steps", None)
        if backend.env == "gameboy" and self.mode == "test":     # GameBoyRL keeps the budget inside its emulator
            from cusi.eval.gameboy import MAX_STEPS
            self.max_steps = MAX_STEPS
        self.done = False
        self.success = None
        self.end_reason = None
        self._frames = []           # Web: the episode's screenshots, for the judge
        self._last = None
        self.recorder = None if record_dir is None else Recorder(directory=record_dir, session_id=self.id, meta={
            "task_id": task_id, "scene": scene, "env": backend.env, "policy": policy.kind, "want_state": want_state,
            "goal": self.goal, "opened": time.time()})

    # ------------------------------------------------------------------ payloads

    def _public(self, *, obs: dict, info: dict, reward, terminated: bool, truncated: bool) -> dict:
        info = {k: v for k, v in info.items() if k not in HIDDEN_INFO + DROPPED_INFO}
        if self.backend.env == "gameboy":     # the supervisor reads the emulator's current frame
            info["current_frame"] = self.env._env.get_info()["core"]["current_frame"]
        out = {"session": self.id, "obs": encode(obs), "info": encode(info), "reward": reward,
               "terminated": terminated, "truncated": truncated, "done": self.done, "success": self.success,
               "end_reason": self.end_reason, "steps": getattr(self.env, "_steps", None),
               "max_steps": self.max_steps}
        if self.want_state and self.policy.state:
            out["state"] = encode(self.env.full_state())
        return out

    def _record(self, *, kind: str, action, obs: dict, info: dict, reward) -> None:
        if self.recorder is not None:
            self.recorder.write(kind=kind, action=action, obs=obs, info=info, reward=reward, done=self.done,
                                success=self.success, state=self.env.full_state())

    def describe(self) -> dict:
        out = {"session": self.id, "env": self.backend.env, "mode": self.mode, "task_id": self.task_id,
               "scene": self.scene, "goal": self.goal, "max_steps": self.max_steps,
               "state_enabled": self.want_state and self.policy.state, "save_load": self.policy.save_load,
               "done": self.done, "success": self.success, "end_reason": self.end_reason,
               "steps": getattr(self.env, "_steps", None)}
        if self.backend.env == "gameboy":
            out["game"] = self.env.game
            out["action_strings"] = self.env.action_strings()
        if self.backend.env == "web":
            out["start_url"] = self.env.start_url
        if self.backend.env == "android":
            out["action_mode"] = self.env.action_mode
            out["extra_action_types"] = list(self.env.extra_action_types)
        return out

    # ------------------------------------------------------------------ actions

    def reset(self) -> dict:
        obs, info = self.env.reset()
        self.done, self.success, self.end_reason = False, None, None
        self._frames = [obs["frame"]] if self.backend.env == "web" else []
        self._last = (obs, info)
        self._record(kind="reset", action=None, obs=obs, info=info, reward=0.0)
        return self._public(obs=obs, info=info, reward=0.0, terminated=False, truncated=False)

    def step(self, *, action: str) -> dict:
        if self.done:
            raise SessionError(f"the episode is over ({self.end_reason}); reset or close the session", 409)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.backend.env == "web":
            self._frames.append(obs["frame"])
        if self.policy.kind == "dev" and self._held_out(info=info):
            # the env undoes steps into held-out apps and sites (its blocked lists); still there means the undo failed
            self.done, self.end_reason = True, "held_out_reached"
            log_warn(f"[env_server] {self.id}: still in a held-out app/site after a step: session ended",
                     parameters=self._parameters)
        elif terminated or truncated:
            self.done = True
            self.end_reason = "terminated" if terminated else "truncated"
            if self.mode == "test":
                self.success = self._success(reward=reward, terminated=terminated, info=info)
        self._last = (obs, info)
        self._record(kind="step", action=action, obs=obs, info=info, reward=reward)
        return self._public(obs=obs, info=info, reward=reward, terminated=terminated, truncated=truncated)

    def observe(self) -> dict:
        """The current screen without acting: Android re-observes the device (M3A does before each decision)."""
        if self.backend.env == "android":
            obs, info = self.env.observe()
            self._last = (obs, info)
        obs, info = self._last
        return self._public(obs=obs, info=info, reward=0.0, terminated=False, truncated=False)

    def context(self) -> dict:
        """Android: the foreground app and the device date/time, as M3A's prompt shows them."""
        if self.backend.env != "android":
            return {"context": None}
        from cusi.agents.executors.m3a import device_context
        return {"context": device_context(env=self.env)}

    def save(self) -> dict:
        if not self.policy.save_load:
            raise SessionError("save/load is not available on this server", 403)
        return {"handle": self.env.save_state()}

    def load(self, *, handle: str) -> dict:
        if not self.policy.save_load:
            raise SessionError("save/load is not available on this server", 403)
        self.env.load_state(state_id=handle)
        obs = self.env.current_obs
        info = {"loaded": handle}
        self.done, self.success, self.end_reason = False, None, None
        self._last = (obs, info)
        self._record(kind="load", action=handle, obs=obs, info=info, reward=0.0)
        return self._public(obs=obs, info=info, reward=0.0, terminated=False, truncated=False)

    def close(self) -> None:
        self.backend.release(env=self.env)

    # ------------------------------------------------------------------ checks

    def _held_out(self, *, info: dict) -> bool:
        """Dev: the env is on a held-out website or app after a step (a failed undo; the safety net)."""
        if self.backend.env == "web":
            from cusi.envs.webvoyager import in_scope
            url = info.get("url") or (info.get("true") or {}).get("url")
            return bool(url) and url.startswith("http") and any(in_scope(url=url, scope=s) for s in held_out_scopes())
        if self.backend.env == "android":
            activity = (info.get("true") or {}).get("activity") or ""
            return activity.split("/")[0] in held_out_packages()
        return False

    def _success(self, *, reward, terminated: bool, info: dict) -> Optional[float]:
        if not terminated:
            return 0.0
        if self.backend.env == "web":
            answer = info.get("answer")
            if answer is None:
                return 0.0
            verdict = self.backend.judge_web(goal=self.goal, answer=answer, frames=self._frames)
            log_info(f"[env_server] web judge {self.task_id}: {verdict}", parameters=self._parameters)
            return verdict
        return float(reward or 0.0)


class Sessions:
    """The server's live sessions, capped, closed after idle_seconds without a request."""

    def __init__(self, *, backend: Backend, policy: Policy, max_sessions: int, idle_seconds: float,
                 record_dir: Optional[str], parameters: dict) -> None:
        self.backend, self.policy = backend, policy
        self.max_sessions, self.idle_seconds, self.record_dir = max_sessions, idle_seconds, record_dir
        self._parameters = parameters
        self._sessions: dict = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._reap, daemon=True).start()

    def open(self, *, task_id: Optional[str], scene: Optional[str], want_state: bool) -> Session:
        if (task_id is None) == (scene is None):
            raise SessionError("open a session with exactly one of task_id or scene")
        if task_id is not None:
            if not self.policy.visible(task_id=task_id, parameters=self._parameters):
                raise SessionError(f"unknown task {task_id!r}", 404)
            if env_of(task_id=task_id) != self.backend.env:
                raise SessionError(f"{task_id!r} is not a {self.backend.env} task (this server serves "
                                   f"{self.backend.env})", 404)
        else:
            if not self.policy.free_play:
                raise SessionError("free play is not available on this server", 403)
            if scene not in self.backend.free_play_scenes():
                raise SessionError(f"unknown scene {scene!r}", 404)
        if want_state and not self.policy.state:
            raise SessionError("state is not available on this server", 403)
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise SessionError(f"at most {self.max_sessions} sessions at once; close one first", 429)
            placeholder = object()
            key = uuid.uuid4().hex
            self._sessions[key] = placeholder     # holds the slot while the env builds
        try:
            session = Session(backend=self.backend, policy=self.policy, task_id=task_id, scene=scene,
                              want_state=want_state, record_dir=self.record_dir, parameters=self._parameters)
        except Exception:
            with self._lock:
                del self._sessions[key]
            raise
        with self._lock:
            del self._sessions[key]
            self._sessions[session.id] = session
        log_info(f"[env_server] opened {session.id}: {task_id or scene}", parameters=self._parameters)
        return session

    def token_of(self, *, session_id: str) -> Optional[str]:
        with self._lock:
            session = self._sessions.get(session_id)
        return session.token if isinstance(session, Session) else None

    def get(self, *, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if not isinstance(session, Session):
            raise SessionError(f"no session {session_id!r}", 404)
        session.last_used = time.time()
        return session

    def close(self, *, session_id: str) -> None:
        session = self.get(session_id=session_id)
        with self._lock:
            self._sessions.pop(session_id, None)
        with session.lock:
            session.close()
        log_info(f"[env_server] closed {session_id}", parameters=self._parameters)

    def close_all(self) -> None:
        for sid in [k for k, v in self._sessions.items() if isinstance(v, Session)]:
            try:
                self.close(session_id=sid)
            except Exception as e:
                log_warn(f"[env_server] closing {sid}: {e}", parameters=self._parameters)

    def _reap(self) -> None:
        while True:
            time.sleep(30)
            now = time.time()
            idle = [k for k, v in list(self._sessions.items())
                    if isinstance(v, Session) and now - v.last_used > self.idle_seconds]
            for sid in idle:
                log_warn(f"[env_server] {sid} idle for {self.idle_seconds:.0f}s: closing", parameters=self._parameters)
                try:
                    self.close(session_id=sid)
                except Exception as e:
                    log_warn(f"[env_server] closing {sid}: {e}", parameters=self._parameters)
