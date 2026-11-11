"""
Client for the environment servers. Standard library only; frames come back as PNG bytes (to_array() turns one into
a numpy array when numpy and Pillow are installed).

Servers are configured in envs.json next to this file (or $ENVCLIENT_CONFIG):
    {"token_file": "...", "servers": {"gameboy": "http://127.0.0.1:8710", "web": "...", "android": "..."}}
or by $ENV_SERVER_<ENV> and $ENV_SERVER_TOKEN. In an evaluation, $ENVCTL_DEADLINE is the unix time the task's
wall-clock limit ends (seconds_left() counts down to it).

    from envclient import Client
    c = Client.from_config()
    s = c.open(task_id="gb.pokemon_red.004")            # or c.open(env="gameboy", scene="pokemon_red/<state>")
    out = s.step("A")                                    # out["obs"]["frame"] is PNG bytes; out["done"], out["success"]
    s.close()
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request

PREFIX_ENV = {"gb": "gameboy", "web": "web", "aw": "android"}


def seconds_left():
    """Seconds until the task's wall-clock limit ($ENVCTL_DEADLINE), or None outside an evaluation."""
    deadline = os.environ.get("ENVCTL_DEADLINE")
    return None if not deadline else max(0.0, float(deadline) - time.time())


class EnvError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status


def decode(value):
    """Server markers back to Python values: {"__png__"} -> PNG bytes, {"__array__"} -> nested list, enums and classes
    -> their names."""
    if isinstance(value, dict):
        if "__png__" in value:
            return base64.b64decode(value["__png__"])
        if "__array__" in value:
            return value["__array__"]
        if "__enum__" in value:
            return value["__enum__"].split(":", 1)[1]
        if "__class__" in value:
            return value["__class__"].split(":", 1)[1]
        return {k: decode(v) for k, v in value.items() if k != "__obj__"}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def to_array(png: bytes):
    import io
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(png)))


def env_of(task_id: str) -> str:
    return PREFIX_ENV[task_id.split(".", 1)[0]]


class Server:
    def __init__(self, url: str, token: str, timeout: float = 900.0) -> None:
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def request(self, method: str, path: str, body: dict = None, raw: bool = False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", "")
            except Exception:
                msg = str(e)
            raise EnvError(e.code, msg) from None
        return out if raw else decode(out)


class Session:
    def __init__(self, server: Server, opened: dict) -> None:
        self.server = server
        self.id = opened["session"]
        self.info = {k: opened[k] for k in ("env", "mode", "task_id", "scene", "goal", "max_steps") if k in opened}
        self.first = opened

    def _post(self, action: str, body: dict = None, raw: bool = False):
        return self.server.request("POST", f"/sessions/{self.id}/{action}", body or {}, raw=raw)

    def step(self, action: str, raw: bool = False) -> dict:
        return self._post("step", {"action": action}, raw=raw)

    def reset(self, raw: bool = False) -> dict:
        return self._post("reset", raw=raw)

    def observe(self, raw: bool = False) -> dict:
        return self._post("observe", raw=raw)

    def context(self):
        return self.server.request("GET", f"/sessions/{self.id}/context")["context"]

    def save(self) -> str:
        return self._post("save")["handle"]

    def load(self, handle: str, raw: bool = False) -> dict:
        return self._post("load", {"handle": handle}, raw=raw)

    def describe(self) -> dict:
        return self.server.request("GET", f"/sessions/{self.id}")

    def close(self) -> None:
        self.server.request("DELETE", f"/sessions/{self.id}")


class Client:
    def __init__(self, servers: dict, token: str) -> None:
        self.servers = {env: Server(url, token) for env, url in servers.items()}

    @classmethod
    def from_config(cls, path: str = None) -> "Client":
        path = path or os.environ.get("ENVCLIENT_CONFIG") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                         "envs.json")
        servers, token = {}, os.environ.get("ENV_SERVER_TOKEN")
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            servers.update(cfg.get("servers", {}))
            if token is None and cfg.get("token_file"):
                with open(cfg["token_file"]) as f:
                    token = f.read().strip()
            token = token or cfg.get("token")
        for env in PREFIX_ENV.values():
            if os.environ.get(f"ENV_SERVER_{env.upper()}"):
                servers[env] = os.environ[f"ENV_SERVER_{env.upper()}"]
        if not servers or not token:
            raise RuntimeError(f"no servers or token configured (looked at {path} and $ENV_SERVER_*)")
        return cls(servers=servers, token=token)

    def server(self, env: str) -> Server:
        if env not in self.servers:
            raise EnvError(404, f"no {env} server configured (have {sorted(self.servers)})")
        return self.servers[env]

    def tasks(self, env: str = None) -> list:
        envs = [env] if env else list(self.servers)
        return [t for e in envs for t in self.server(e).request("GET", "/tasks")["tasks"]]

    def scenes(self, env: str) -> list:
        return self.server(env).request("GET", "/scenes")["scenes"]

    def open(self, task_id: str = None, env: str = None, scene: str = None, state: bool = False,
             raw: bool = False) -> Session:
        env = env_of(task_id) if task_id else env
        server = self.server(env)
        body = {"task_id": task_id} if task_id else {"scene": scene}
        body["state"] = state
        return Session(server, server.request("POST", "/sessions", body, raw=raw))

    def attach(self, env: str, session_id: str) -> Session:
        """A session opened elsewhere (by the test harness, or an earlier envctl call)."""
        server = self.server(env)
        return Session(server, {"session": session_id, **server.request("GET", f"/sessions/{session_id}")})
