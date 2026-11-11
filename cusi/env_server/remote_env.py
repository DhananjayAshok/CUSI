"""
RemoteEnv: an env-like proxy over an env server session (user_template/envclient.py), so CUSI's agents
(cusi/agents/agent.py) run through a server exactly as they run on a local env: same obs, same info, the server's
step budget and success.

    env = RemoteEnv.open(url=..., token=..., task_id="gb.pokemon_red.004")
    obs, info = env.first
    agent = make_agent(env_name=env.env_name, env=env, vlm=vlm, parameters=parameters)
"""
import importlib.util
import os
from typing import Optional
from cusi.env_server.codec import decode


def _envclient():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "user_template", "envclient.py")
    spec = importlib.util.spec_from_file_location("envclient", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CoreShim:
    """GameBoy's SubgoalSupervisor reads env._env.get_info()["core"]["current_frame"]; the server sends that frame."""

    def __init__(self, env: "RemoteEnv") -> None:
        self._remote = env

    def get_info(self) -> dict:
        return {"core": {"current_frame": self._remote.last_info.get("current_frame")}}


class RemoteEnv:
    def __init__(self, *, session, opened: dict) -> None:
        self._session = session
        d = decode(opened)
        self.env_name = d["env"]
        self.mode = d["mode"]
        self.task_id = d.get("task_id")
        self.goal = d.get("goal")
        self.max_steps = d.get("max_steps")
        self.game = d.get("game", "")
        self._action_strings = d.get("action_strings")
        self.start_url = d.get("start_url")
        self.action_mode = d.get("action_mode", "all")
        self.extra_action_types = tuple(d.get("extra_action_types") or ())
        self._work_dir = None       # downloaded PDFs stay on the server: WebVoyager's PDF answer falls back to text
        self._env = _CoreShim(self) if self.env_name == "gameboy" else None
        self.last = d
        self.current_obs, self.last_info = d["obs"], d["info"]
        self.first = (d["obs"], d["info"])

    @classmethod
    def open(cls, *, url: str, token: str, task_id: Optional[str] = None, env: Optional[str] = None,
             scene: Optional[str] = None, state: bool = False) -> "RemoteEnv":
        ec = _envclient()
        server = ec.Server(url, token)
        env = env or ec.env_of(task_id)
        body = {"task_id": task_id} if task_id else {"scene": scene}
        body["state"] = state
        opened = server.request("POST", "/sessions", body, raw=True)
        return cls(session=ec.Session(server, {"session": opened["session"]}), opened=opened)

    def _take(self, out: dict) -> dict:
        d = decode(out)
        if self.env_name == "gameboy":    # the server never sends GameBoyRL's transition records (they hold RAM state)
            d["info"].setdefault("transition_states", [])
        self.last = d
        self.current_obs, self.last_info = d["obs"], d["info"]
        return d

    @property
    def raw_frame(self):
        return self.last_info.get("raw_frame", self.current_obs.get("frame"))

    def reset(self, **_):
        d = self._take(self._session.reset(raw=True))
        return d["obs"], d["info"]

    def step(self, action: str):
        d = self._take(self._session.step(action, raw=True))
        return d["obs"], d["reward"], d["terminated"], d["truncated"], d["info"]

    def observe(self):
        d = self._take(self._session.observe(raw=True))
        return d["obs"], d["info"]

    def action_strings(self, *, return_all: bool = False) -> dict:
        return self._action_strings

    def remote_device_context(self) -> str:
        return self._session.context()

    @property
    def done(self) -> bool:
        return bool(self.last.get("done"))

    @property
    def success(self):
        return self.last.get("success")

    def close(self) -> None:
        self._session.close()
