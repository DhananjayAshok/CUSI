"""
Evaluate a User's workspace on a schedule of test tasks (cusi/env_server/harness.py): sessions on test env servers,
the User run in the sandbox with only that session, scores from the servers.

    python run_user_eval.py --workspace <dir> --servers '{"gameboy": "http://127.0.0.1:8720"}' \
        --token_file <test servers' token> --tasks gb.pokemon_red.021,gb.pokemon_red.025 --out <dir> \
        --mode fixed_repo --memory carry --wall_clock_seconds 1800 [--order shuffle --seed 0]

--mode fixed_repo runs the workspace's policy.json entrypoint; live_agent starts prime-agent (--agent_home holds its
install and login). --memory carry: one workspace through the schedule; isolated: every task from the same
snapshot. --order: given (as listed), shuffle, group_env (environments in sequence, tasks shuffled within each).
"""
import json
import click
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info
from cusi.eval.options import task_list
from cusi.env_server.harness import MEMORY, MODES, ORDERS, Harness, order_tasks
from cusi.env_server.sandbox import agent_home, sandbox_image


@click.command()
@click.option("--workspace", required=True, help="The User's workspace (its state at the start of the schedule).")
@click.option("--out", required=True, help="Output directory (resumes a schedule already there).")
@click.option("--mode", required=True, type=click.Choice(MODES))
@click.option("--memory", required=True, type=click.Choice(MEMORY))
@click.option("--servers", required=True, help='JSON {env: url} of the test servers.')
@click.option("--token_file", required=True, help="The test servers' token (never given to the User).")
@click.option("--tasks", required=True, help="Task ids, comma-separated or a file of lines.")
@click.option("--order", default="given", show_default=True, type=click.Choice(ORDERS))
@click.option("--seed", default=0, show_default=True, help="For --order shuffle / group_env.")
@click.option("--wall_clock_seconds", required=True, type=float, help="Wall-clock limit per task.")
@click.option("--agent_home", "agent_home_dir", default=None,
              help="live_agent: the agent home to start from (default storage_dir/env_server/agent_home).")
@click.option("--agent_cmd", default=None, help="live_agent: the agent's command (default prime-agent).")
@click.option("--image", default=None, help="Sandbox image (default storage_dir/containers/sandbox.sif).")
def main(workspace, out, mode, memory, servers, token_file, tasks, order, seed, wall_clock_seconds, agent_home_dir,
         agent_cmd, image):
    parameters = load_parameters()
    with open(token_file) as f:
        token = f.read().strip()
    schedule = order_tasks(tasks=task_list(tasks=tasks).split(","), order=order, seed=seed)
    if mode == "live_agent" and agent_home_dir is None:
        agent_home_dir = agent_home(parameters=parameters)
    harness = Harness(workspace=workspace, out=out, mode=mode, memory=memory, servers=json.loads(servers), token=token,
                      wall_clock_seconds=wall_clock_seconds, image=image or sandbox_image(parameters=parameters),
                      parameters=parameters, agent_home=agent_home_dir, agent_cmd=agent_cmd)
    rows = harness.run(schedule=schedule, meta={"order": order, "seed": seed, "servers": json.loads(servers)})
    log_info(f"[user_eval] {len(rows)} tasks run -> {out}", parameters=parameters)


if __name__ == "__main__":
    main()
