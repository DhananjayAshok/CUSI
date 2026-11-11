"""
End-to-end check of a running env server: CUSI's final agent for the env (cusi/agents/agent.py) solves tasks through
the server (RemoteEnv) instead of a local env, and the server's answers are checked against its policy.

    python -m cusi.env_server.check --env gameboy --url http://127.0.0.1:8710 --token_file <f> \
        --model google/gemma-4-31b-it --vllm_base_url http://127.0.0.1:8100/v1/ --tasks gb.pokemon_red.004

One env per process (GameBoyRL's and WebVoyager's `utils` modules cannot share one). Prints one JSON line per task.
"""
import json
import time
import click
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(["gameboy", "web", "android"]))
@click.option("--url", required=True)
@click.option("--token_file", required=True)
@click.option("--model", required=True, help="The agent's model, as served.")
@click.option("--vllm_base_url", required=True)
@click.option("--tasks", required=True, help="Comma-separated task ids.")
@click.option("--max_steps", default=None, type=int, help="Cap the agent's budget (default: the server's).")
def main(env_name, url, token_file, model, vllm_base_url, tasks, max_steps):
    from cusi.agents.agent import make_agent
    from cusi.eval.supervision import model as make_model
    from cusi.env_server.remote_env import RemoteEnv
    parameters = load_parameters()
    with open(token_file) as f:
        token = f.read().strip()
    vlm = make_model(model_name=model, model_backend="vllm", vllm_base_url=vllm_base_url)
    for task_id in tasks.split(","):
        t0 = time.time()
        env = RemoteEnv.open(url=url, token=token, task_id=task_id)
        row = {"env": env_name, "task_id": task_id, "goal": env.goal, "budget": env.max_steps}
        try:
            obs, info = env.first
            agent = make_agent(env_name=env_name, env=env, vlm=vlm, parameters=parameters)
            budget = min(max_steps, env.max_steps) if max_steps and env.max_steps else (max_steps or env.max_steps)
            _, result = agent.run(task=env.goal, obs=obs, info=info, max_steps=budget)
            report = result["report"]
            final = report.last_report
            row.update(agent_steps=report.n_steps, n_invalid=report.n_invalid,
                       termination=final.termination_reason if final else None,
                       answer=final.answer if final else None,
                       server_done=env.done, server_success=env.success, server_steps=env.last.get("steps"),
                       server_end=env.last.get("end_reason"),
                       state_in_responses="state" in env.last, true_in_info="true" in env.last_info)
        except Exception as e:
            import traceback
            row["error"] = traceback.format_exc()[-1500:]
        finally:
            env.close()
        row["seconds"] = round(time.time() - t0, 1)
        print("CHECK " + json.dumps(row, default=str), flush=True)
        log_info(f"[env_server check] {task_id} done", parameters=parameters)


if __name__ == "__main__":
    main()
