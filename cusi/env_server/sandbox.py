"""
The User's sandbox: an Apptainer container (--containall) that mounts only the agent's workspace at /work and, as
its home, an agent home of its own (storage_dir/env_server/agent_home: prime-agent's install, login and sessions).
The host's home, CUSI, the rest of storage, GameBoyRL and the Hugging Face cache are not mounted; the network is the
host's (env servers on 127.0.0.1, API models, the open web).

The image (storage_dir/containers/sandbox.sif, docker://python:3.12: Debian, Python, curl, git) has nothing of ours:
the client (user_template/envclient.py, envctl) is standard library only and lives in the workspace.
"""
import os
import shutil
import subprocess
from typing import Optional

WORKDIR = "/work"
HOME = "/home/agent"
PATH = f"{HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin"


def agent_home(*, parameters: dict) -> str:
    path = os.path.join(parameters["storage_dir"], "env_server", "agent_home")
    os.makedirs(path, exist_ok=True)
    return path


def sandbox_image(*, parameters: dict) -> str:
    return os.path.join(parameters["storage_dir"], "containers", "sandbox.sif")


def sandbox_argv(*, image: str, workspace: str, cmd: list, env_vars: Optional[dict] = None,
                 read_only: bool = False, home: Optional[str] = None, interactive: bool = False,
                 binds: Optional[list] = None) -> list:
    """apptainer exec argv running cmd in /work (the workspace) with only env_vars set; home: a host directory
    mounted as the sandbox's $HOME (default: an empty one that does not outlive the call); binds: further
    "src:dst[:ro]" mounts."""
    argv = ["apptainer", "exec", "--containall", "--cleanenv", "--bind",
            f"{os.path.abspath(workspace)}:{WORKDIR}{':ro' if read_only else ''}", "--pwd", WORKDIR,
            "--env", f"PATH={PATH}"]
    if home is not None:
        argv += ["--home", f"{os.path.abspath(home)}:{HOME}"]
    for b in binds or []:
        argv += ["--bind", b]
    if interactive:
        argv += ["--env", f"TERM={os.environ.get('TERM', 'xterm-256color')}"]
    for k, v in (env_vars or {}).items():
        argv += ["--env", f"{k}={v}"]
    return argv + [image] + list(cmd)


def run_in_sandbox(*, image: str, workspace: str, cmd: list, env_vars: Optional[dict] = None, timeout: float = None,
                   log_path: Optional[str] = None, home: Optional[str] = None, binds: Optional[list] = None) -> int:
    """Run cmd in the sandbox; stdout and stderr go to log_path (or are inherited). Returns the exit code (124 on
    timeout)."""
    if shutil.which("apptainer") is None:
        raise RuntimeError("apptainer is not on PATH")
    argv = sandbox_argv(image=image, workspace=workspace, cmd=cmd, env_vars=env_vars, home=home, binds=binds)
    out = open(log_path, "w") if log_path else None
    try:
        return subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT if out else None, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124
    finally:
        if out:
            out.close()


def make_workspace(*, template: str, directory: str, servers: dict, token: str) -> str:
    """A fresh workspace from user_template: the client, envs.json (servers and the token) and the template's files.
    The token goes into envs.json itself: the sandbox cannot read a token file elsewhere."""
    shutil.copytree(template, directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(directory, "envs.json"), "w") as f:
        import json
        json.dump({"token": token, "servers": servers}, f, indent=1)
    os.chmod(os.path.join(directory, "envctl"), 0o755)
    return directory


def main() -> None:
    """Open the sandbox on a workspace (made from user_template if it does not exist) and run a command in it,
    interactively, with the agent home as $HOME:

        python -m cusi.env_server.sandbox --workspace <dir> [--servers '{"gameboy": "http://..."}'
            --token_file <dev token>] [-- prime-agent ...]          (default command: bash)
    """
    import argparse
    import json
    import sys
    from cusi.utils.parameter_handling import load_parameters
    ap = argparse.ArgumentParser(prog="python -m cusi.env_server.sandbox")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--servers", default=None, help="JSON {env: url}; written into a new workspace's envs.json")
    ap.add_argument("--token_file", default=None, help="The dev servers' token, for a new workspace's envs.json")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    parameters = load_parameters()
    if not os.path.isdir(a.workspace):
        token = open(a.token_file).read().strip() if a.token_file else ""
        make_workspace(template=os.path.join(parameters["project_root"], "user_template"), directory=a.workspace,
                       servers=json.loads(a.servers) if a.servers else {}, token=token)
        subprocess.run(["git", "init", "-q", a.workspace])
    cmd = [c for c in a.cmd if c != "--"] or ["bash"]
    argv = sandbox_argv(image=sandbox_image(parameters=parameters), workspace=a.workspace, cmd=cmd,
                        home=agent_home(parameters=parameters), interactive=True)
    sys.exit(subprocess.run(argv).returncode)


if __name__ == "__main__":
    main()
