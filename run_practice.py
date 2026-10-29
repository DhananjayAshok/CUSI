"""
The practice pipeline: attempt tasks (judged, with critique-hint retries) and build the SFT datasets from the
attempts. The tasks come from an annotated search run (--source <run>: every node's tasks, each started from its
node restored).

    python run_practice.py --env gameboy --model_name <served name> --source <annotated search run> all
"""
import json
import os
import click
from cusi.utils.parameter_handling import load_parameters, compute_secondary_parameters
from cusi.utils.log_handling import log_info
from cusi.agents.specs import ENV_NAMES, ENV_SPECS, EnvPool
from cusi.utils.paths import PracticePaths, search_run
from cusi.search.restore import DEFAULT_STATE_CACHE
from cusi.agents.vlm import AgentVLM

loaded_parameters = load_parameters()


@click.group()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--model_name", required=True, help="Model name the vLLM server serves; used for every role.")
@click.option("--model_backend", default="vllm")
@click.option("--max_new_tokens", default=2000, help="Token budget for judge/critique/paraphrase calls.")
@click.option("--overwrite", is_flag=True, default=False, help="Redo the dataset stage.")
@click.option("--random_seed", default=loaded_parameters["random_seed"])
@click.option("--vllm_port", default=loaded_parameters["vllm_port"], help="Port of the vLLM server to use.")
@click.option("--source", required=True, help="An annotated search run of this env (cusi.search.annotate).")
@click.option("--state_cache", default=None, type=int,
              help="Replay restores: keep saved states of this many recent nodes and replay from the nearest one "
                   "(default: 64 on Android, 0 elsewhere).")
@click.pass_context
def main(ctx, env_name, model_name, model_backend, max_new_tokens, overwrite, random_seed, vllm_port, source,
         state_cache):
    loaded_parameters["random_seed"] = random_seed
    loaded_parameters["vllm_port"] = vllm_port
    compute_secondary_parameters(loaded_parameters)
    spec = ENV_SPECS[env_name]
    ctx.obj = {
        "parameters": loaded_parameters, "spec": spec, "overwrite": overwrite, "max_new_tokens": max_new_tokens,
        "paths": PracticePaths(parameters=loaded_parameters, env_name=env_name, model_name=model_name, source=source),
        "source": source, "env_name": env_name,
        "state_cache": DEFAULT_STATE_CACHE[env_name] if state_cache is None else state_cache,
        "vlm": AgentVLM(model_name=model_name, model_backend=model_backend, parameters=loaded_parameters),
    }
    log_info(f"run_practice: env={env_name} model={model_name} source={source} -> {ctx.obj['paths'].root}",
             parameters=loaded_parameters)


def _pool(obj) -> EnvPool:
    if "pool" not in obj:
        # As the source tree's search: scenes kept on their website or app only if it kept them (run_search).
        with open(os.path.join(search_run(parameters=obj["parameters"], env=obj["env_name"], run=obj["source"]),
                               "config.json")) as f:
            config = json.load(f)
        scoped = bool(config.get("config", config).get("scoped", False))
        obj["pool"] = EnvPool(spec=obj["spec"], parameters=obj["parameters"], scoped=scoped)
    return obj["pool"]


def _close(obj) -> None:
    if "pool" in obj:
        obj["pool"].close()


def do_attempt(obj, max_tries, lookback, max_tasks):
    from cusi.practice.stages.attempt import attempt
    from cusi.search.restore import NodeRestorer
    restorer = NodeRestorer(directory=search_run(parameters=obj["parameters"], env=obj["env_name"],
                                                 run=obj["source"]), parameters=obj["parameters"],
                            state_cache=obj["state_cache"])
    attempt(spec=obj["spec"], pool=_pool(obj), vlm=obj["vlm"], paths=obj["paths"], restorer=restorer,
            max_tries=max_tries, lookback=lookback, judge_max_new_tokens=obj["max_new_tokens"], max_tasks=max_tasks,
            parameters=obj["parameters"])


def do_dataset(obj, n_paraphrases, safety_margin, val_frac, seed):
    from cusi.practice.stages.dataset import build_dataset
    build_dataset(spec=obj["spec"], vlm=obj["vlm"], paths=obj["paths"], n_paraphrases=n_paraphrases,
                  safety_margin=safety_margin, val_frac=val_frac, seed=seed, max_new_tokens=obj["max_new_tokens"],
                  overwrite=obj["overwrite"], parameters=obj["parameters"])


max_tries_opt = click.option("--max_tries", default=3, help="Tries per task (a critique hint between tries).")
lookback_opt = click.option("--lookback", default=8, help="Final frames shown to the judge.")
max_tasks_opt = click.option("--max_tasks", default=None, type=int, help="Attempt only the first this many tasks.")
paraphrases_opt = click.option("--n_paraphrases", default=3, help="Paraphrases per successful train task.")
margin_opt = click.option("--safety_margin", default=2, help="Executor calls kept past the safe success point.")
val_opt = click.option("--val_frac", default=0.2, help="Share of tasks held out for validation.")
seed_opt = click.option("--seed", default=0, help="Train/validation task split seed.")


@main.command()
@max_tries_opt
@lookback_opt
@max_tasks_opt
@click.pass_obj
def attempt(obj, max_tries, lookback, max_tasks):
    """Attempt each task, judged, with critique-hint retries; one pickle per task."""
    try:
        do_attempt(obj, max_tries, lookback, max_tasks)
    finally:
        _close(obj)


@main.command()
@paraphrases_opt
@margin_opt
@val_opt
@seed_opt
@click.pass_obj
def dataset(obj, n_paraphrases, safety_margin, val_frac, seed):
    """Build the success and failure train/validation JSONL from the attempts."""
    do_dataset(obj, n_paraphrases, safety_margin, val_frac, seed)


@main.command(name="all")
@max_tries_opt
@lookback_opt
@max_tasks_opt
@paraphrases_opt
@margin_opt
@val_opt
@seed_opt
@click.pass_obj
def run_all(obj, max_tries, lookback, max_tasks, n_paraphrases, safety_margin, val_frac, seed):
    """attempt, dataset."""
    try:
        do_attempt(obj, max_tries, lookback, max_tasks)
        do_dataset(obj, n_paraphrases, safety_margin, val_frac, seed)
    finally:
        _close(obj)


if __name__ == "__main__":
    main()
