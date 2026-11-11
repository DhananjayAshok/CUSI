"""
The evaluation harness (plans/user_sandbox_plan.md): a schedule of test tasks, run in order; for each, open a session
on a test env server, run the User in the sandbox with only that session, score from the server.

Modes: fixed_repo runs the command the workspace's policy.json names ({"entrypoint": "..."}); live_agent starts
prime-agent with a prompt (prompts/live_agent.md) and the agent home. Budgets, both modes: the task's step budget
(the server's) and a wall-clock limit per task (the harness ends the process).

Memory: carry keeps one workspace (and, live_agent, one agent home) through the schedule, so what task k wrote is
there at task k+1; isolated starts every task from the same snapshot. Nothing is discarded: every task's workspace
is committed (carry: one repo, a commit per task; isolated: one repo per task).

    <out>/schedule.json      the schedule, mode, memory, snapshot commit, budgets
    <out>/results.jsonl      one line per task, in schedule order (resumed runs skip finished ones)
    <out>/workspace/         carry: the evolving workspace (git: a commit per task)
    <out>/runs/<k>_<task>/   isolated: that task's workspace (git)
    <out>/agent_home/        live_agent + carry: the evolving agent memory (~/.prime, ~/.config; not in git: its
                             session store holds a Python environment)
    <out>/logs/<k>_<task>.log
"""
import importlib.util
import json
import os
import random
import shlex
import shutil
import subprocess
import time
from typing import Optional
from cusi.utils.log_handling import log_error, log_info, log_warn
from cusi.env_server.sandbox import HOME, run_in_sandbox

MODES = ("fixed_repo", "live_agent")
MEMORY = ("isolated", "carry")
ORDERS = ("given", "shuffle", "group_env")
#: Never copied from the User's workspace into an evaluation (development-time session files and tokens).
SKIPPED = (".envctl", "obs", "envs.json", "__pycache__")
#: The agent home's memory (copied per task or carried); the rest (the install, caches) is shared read-only.
AGENT_MEMORY = (".prime", ".config")
MEMORY_NOTES = {"carry": "Tasks come one after another: what you write in /work (and your own memory) is still there at the next task.",
                "isolated": "Every task starts from the same workspace: what you write now is not kept for the next task."}
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "live_agent.md")


def _client(*, parameters: dict):
    spec = importlib.util.spec_from_file_location(
        "envclient", os.path.join(parameters["project_root"], "user_template", "envclient.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe(task_id: str) -> str:
    """A task id as a file name (WebVoyager ids have spaces: "web.Google Map--0")."""
    return task_id.replace(" ", "_").replace("/", "_")


def _git(*, directory: str, args: list) -> str:
    out = subprocess.run(["git", "-C", directory, *args], capture_output=True, text=True)
    return out.stdout.strip()


def commit_all(*, directory: str, message: str) -> str:
    """Commit everything in directory (a git repo; made one if not) -> the commit hash."""
    if not os.path.isdir(os.path.join(directory, ".git")):
        _git(directory=directory, args=["init", "-q"])
    _git(directory=directory, args=["add", "-A"])
    _git(directory=directory, args=["-c", "user.name=cusi-harness", "-c", "user.email=harness@cusi", "commit", "-q",
                                    "--allow-empty", "-m", message])
    return _git(directory=directory, args=["rev-parse", "HEAD"])


def order_tasks(*, tasks: list, order: str, seed: int) -> list:
    """given: as listed; shuffle: one random order; group_env: environments in first-appearance order, each env's tasks
    shuffled within it (a task schedule per environment, environments in sequence)."""
    rng = random.Random(seed)
    if order == "given":
        return list(tasks)
    if order == "shuffle":
        out = list(tasks)
        rng.shuffle(out)
        return out
    envs = []
    for t in tasks:
        if t.split(".")[0] not in envs:
            envs.append(t.split(".")[0])
    out = []
    for e in envs:
        group = [t for t in tasks if t.split(".")[0] == e]
        rng.shuffle(group)
        out += group
    return out


def _copy_workspace(*, src: str, dst: str) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIPPED), dirs_exist_ok=True)


def _copy_agent_memory(*, src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for name in AGENT_MEMORY:
        if os.path.exists(os.path.join(src, name)):
            shutil.copytree(os.path.join(src, name), os.path.join(dst, name), symlinks=True, dirs_exist_ok=True)


class Harness:
    def __init__(self, *, workspace: str, out: str, mode: str, memory: str, servers: dict, token: str,
                 wall_clock_seconds: float, image: str, parameters: dict, agent_home: Optional[str] = None,
                 agent_cmd: Optional[str] = None) -> None:
        if mode not in MODES or memory not in MEMORY:
            log_error(f"mode must be one of {MODES} and memory one of {MEMORY}", parameters=parameters)
        self.workspace, self.out, self.mode, self.memory = workspace, out, mode, memory
        self.servers, self.token, self.wall_clock, self.image = servers, token, wall_clock_seconds, image
        self.agent_home, self.agent_cmd = agent_home, agent_cmd
        self._parameters = parameters
        self._ec = _client(parameters=parameters)
        if mode == "fixed_repo":
            manifest = os.path.join(workspace, "policy.json")
            if not os.path.exists(manifest):
                log_error(f"fixed_repo needs {manifest} ({{\"entrypoint\": \"<command run in /work>\"}})",
                          parameters=parameters)
            with open(manifest) as f:
                self.entrypoint = json.load(f)["entrypoint"]
        elif agent_home is None or not os.path.isdir(agent_home):
            log_error(f"live_agent needs an agent home (got {agent_home})", parameters=parameters)
        for d in ("logs", "runs"):
            os.makedirs(os.path.join(out, d), exist_ok=True)

    # ---------------------------------------------------------------- per task

    def _workspace_for(self, *, k: int, task_id: str) -> str:
        if self.memory == "carry":
            ws = os.path.join(self.out, "workspace")
            if not os.path.isdir(ws):
                _copy_workspace(src=self.workspace, dst=ws)
                snapshot = commit_all(directory=ws, message="snapshot: the workspace at the start of the schedule")
                with open(os.path.join(self.out, "workspace_snapshot"), "w") as f:
                    f.write(snapshot)
            return ws
        ws = os.path.join(self.out, "runs", f"{k:03d}_{_safe(task_id)}")
        shutil.rmtree(ws, ignore_errors=True)
        _copy_workspace(src=self.workspace, dst=ws)
        return ws

    def _home_for(self, *, k: int, task_id: str) -> Optional[str]:
        if self.mode != "live_agent":
            return None
        if self.memory == "carry":
            home = os.path.join(self.out, "agent_home")
            if not os.path.isdir(home):
                _copy_agent_memory(src=self.agent_home, dst=home)
            return home
        home = os.path.join(self.out, "runs", f"{k:03d}_{_safe(task_id)}.agent_home")
        shutil.rmtree(home, ignore_errors=True)
        _copy_agent_memory(src=self.agent_home, dst=home)
        return home

    def _command(self, *, task: dict) -> list:
        if self.mode == "fixed_repo":
            return ["bash", "-c", self.entrypoint]
        with open(PROMPT_FILE) as f:
            prompt = f.read().format(env=task["env"], max_steps=task["max_steps"],
                                     minutes=round(self.wall_clock / 60, 1), memory_note=MEMORY_NOTES[self.memory])
        cmd = self.agent_cmd or "prime-agent"
        gate = "python3 envctl info | grep -q '\"done\": true'"
        return ["bash", "-c", f"{cmd} --autonomous --autonomous-gate {shlex.quote(gate)} "
                              f"--autonomous-timeout-ms {int(self.wall_clock * 1000)} -p {shlex.quote(prompt)}"]

    def run_task(self, *, k: int, task_id: str) -> dict:
        env = self._ec.env_of(task_id)
        server = self._ec.Server(self.servers[env], self.token)
        t0 = time.time()
        opened = server.request("POST", "/sessions", {"task_id": task_id, "state": False})
        sid = opened["session"]
        ws = self._workspace_for(k=k, task_id=task_id)
        home = self._home_for(k=k, task_id=task_id)
        env_vars = {"ENVCTL_ENV": env, "ENVCTL_SESSION": sid, "ENV_SERVER_TOKEN": opened["session_token"],
                    f"ENV_SERVER_{env.upper()}": self.servers[env]}
        if self.mode == "live_agent":
            env_vars["PRIME_AGENT_TELEMETRY"] = "0"
        log_path = os.path.join(self.out, "logs", f"{k:03d}_{_safe(task_id)}.log")
        binds = None
        if home is not None:      # the install and caches, shared read-only; the memory is the home itself
            binds = [f"{os.path.join(self.agent_home, d)}:{HOME}/{d}:ro" for d in (".local", ".cache")
                     if os.path.isdir(os.path.join(self.agent_home, d))]
        log_info(f"[harness] {k}: {task_id} ({self.mode}, {self.memory}), session {sid}", parameters=self._parameters)
        code = run_in_sandbox(image=self.image, workspace=ws, cmd=self._command(task={"env": env, **opened}),
                              env_vars=env_vars, timeout=self.wall_clock, log_path=log_path, home=home,
                              binds=binds)
        d = server.request("GET", f"/sessions/{sid}")
        try:
            server.request("DELETE", f"/sessions/{sid}")
        except Exception as e:
            log_warn(f"[harness] closing {sid}: {e}", parameters=self._parameters)
        commit = commit_all(directory=ws, message=f"after task {k}: {task_id} (success {d.get('success')})")
        return {"k": k, "task_id": task_id, "env": env, "score": float(d.get("success") or 0.0),
                "success": d.get("success"), "done": d.get("done"), "end_reason": d.get("end_reason"),
                "steps": d.get("steps"), "max_steps": d.get("max_steps"), "exit_code": code,
                "timed_out": code == 124, "seconds": round(time.time() - t0, 1), "workspace_commit": commit}

    # ---------------------------------------------------------------- the schedule

    def _rewind(self, *, finished: list) -> None:
        """carry, resumed: back to the workspace as the last finished task left it (an interrupted task's writes are
        dropped, so a rerun does not see them)."""
        ws = os.path.join(self.out, "workspace")
        if self.memory != "carry" or not os.path.isdir(os.path.join(ws, ".git")):
            return
        if finished:
            target = max(finished, key=lambda r: r["k"])["workspace_commit"]
        else:
            with open(os.path.join(self.out, "workspace_snapshot")) as f:
                target = f.read().strip()
        _git(directory=ws, args=["reset", "-q", "--hard", target])
        _git(directory=ws, args=["clean", "-q", "-fdx"])
        log_info(f"[harness] resumed: workspace back at {target[:8]}", parameters=self._parameters)

    def run(self, *, schedule: list, meta: dict) -> list:
        path = os.path.join(self.out, "schedule.json")
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            if old["schedule"] != schedule or old["mode"] != self.mode or old["memory"] != self.memory:
                log_error(f"{path} holds another schedule/mode/memory: use a new --out", parameters=self._parameters)
        else:
            with open(path, "w") as f:
                json.dump({"schedule": schedule, "mode": self.mode, "memory": self.memory,
                           "snapshot_commit": _git(directory=self.workspace, args=["rev-parse", "HEAD"]) or None,
                           "wall_clock_seconds": self.wall_clock, **meta}, f, indent=1)
        results_path = os.path.join(self.out, "results.jsonl")
        finished = []
        if os.path.exists(results_path):
            with open(results_path) as f:
                finished = [json.loads(l) for l in f if l.strip()]
        done = {r["k"] for r in finished}
        self._rewind(finished=finished)
        rows = []
        for k, task_id in enumerate(schedule):
            if k in done:
                continue
            row = self.run_task(k=k, task_id=task_id)
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            log_info(f"[harness] {json.dumps(row)}", parameters=self._parameters)
        return rows
