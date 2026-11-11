"""
The environment server (cusi/env_server): one env kind, one policy (dev or test), on host:port.

    python run_env_server.py --env gameboy --kind dev --port 8710 --token_file <file>
    python run_env_server.py --env web --kind test --port 8711 --token_file <file> \
        --judge_model_name google/gemma-4-31b-it --judge_model_backend vllm --judge_vllm_base_url http://...

Web and Android run inside the CUSI container (scripts/container.sh); Android needs its emulator started.
"""
import click
from cusi.utils.parameter_handling import load_parameters
from cusi.env_server.policy import KINDS, Policy


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(["gameboy", "web", "android"]))
@click.option("--kind", required=True, type=click.Choice(KINDS), help="dev (train + val) or test.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", required=True, type=int)
@click.option("--token_file", required=True, help="A file holding the bearer token clients must send.")
@click.option("--max_sessions", default=4, show_default=True)
@click.option("--idle_seconds", default=1800, show_default=True, help="Close a session after this long unused.")
@click.option("--record_dir", default=None, help="Record every session's steps (true state included) here. Off "
                                                 "by default.")
@click.option("--judge_model_name", default=None, help="Web: WebVoyager's auto_eval judge, for success.")
@click.option("--judge_model_backend", default=None)
@click.option("--judge_vllm_base_url", default=None)
@click.option("--console_port", default=5554, show_default=True, help="Android: the emulator's console port.")
@click.option("--grpc_port", default=8554, show_default=True)
def main(env_name, kind, host, port, token_file, max_sessions, idle_seconds, record_dir, judge_model_name,
         judge_model_backend, judge_vllm_base_url, console_port, grpc_port):
    from cusi.env_server.sessions import Backend, Sessions
    from cusi.env_server.server import serve
    parameters = load_parameters()
    with open(token_file) as f:
        token = f.read().strip()
    judge = None
    if judge_model_name:
        judge = {"model_name": judge_model_name, "model_backend": judge_model_backend,
                 "vllm_base_url": judge_vllm_base_url}
    backend = Backend(env=env_name, parameters=parameters, judge=judge, console_port=console_port,
                      grpc_port=grpc_port)
    sessions = Sessions(backend=backend, policy=Policy.of(kind=kind), max_sessions=max_sessions,
                        idle_seconds=idle_seconds, record_dir=record_dir, parameters=parameters)
    serve(sessions=sessions, host=host, port=port, token=token, parameters=parameters)


if __name__ == "__main__":
    main()
