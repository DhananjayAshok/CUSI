"""
The practice pipeline: propose, attempt, guidance, practice, clean, dataset.

    python run_practice.py --env gameboy --model_name <served name> all
"""
import click
from cusi.utils.parameter_handling import load_parameters, compute_secondary_parameters
from cusi.utils.log_handling import log_info
from cusi.agents.specs import ENV_NAMES, ENV_SPECS, EnvPool
from cusi.practice.stages.common import TASK_SOURCES
from cusi.utils.paths import PracticePaths
from cusi.agents.vlm import AgentVLM

loaded_parameters = load_parameters()


@click.group()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--model_name", required=True, help="Model name the vLLM server serves; used for every role.")
@click.option("--model_backend", default="vllm")
@click.option("--max_new_tokens", default=2000, help="Token budget for proposer/judge/guidance/clean calls.")
@click.option("--executor_max_new_tokens", default=1000,
              help="Token budget for executor calls (M3A's and WebVoyager's wrappers use 1000).")
@click.option("--temperature", default=None, type=float, help="Executor sampling temperature (None: server default).")
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--random_seed", default=loaded_parameters["random_seed"])
@click.option("--vllm_port", default=loaded_parameters["vllm_port"], help="Port of the vLLM server to use.")
@click.option("--source", default="zeroshot", type=click.Choice(TASK_SOURCES),
              help="zeroshot: proposed tasks; curiosity: tasks from run_explore.py tasks (skip propose/attempt).")
@click.pass_context
def main(ctx, env_name, model_name, model_backend, max_new_tokens, executor_max_new_tokens, temperature, overwrite,
         random_seed, vllm_port, source):
    loaded_parameters["random_seed"] = random_seed
    loaded_parameters["vllm_port"] = vllm_port
    compute_secondary_parameters(loaded_parameters)
    spec = ENV_SPECS[env_name]
    ctx.obj = {
        "parameters": loaded_parameters, "spec": spec, "overwrite": overwrite, "max_new_tokens": max_new_tokens,
        "paths": PracticePaths(parameters=loaded_parameters, env_name=env_name, model_name=model_name, source=source),
        "source": source,
        "vlm": AgentVLM(model_name=model_name, model_backend=model_backend, parameters=loaded_parameters),
        "executor_kwargs": {"max_new_tokens": executor_max_new_tokens, "temperature": temperature},
    }
    log_info(f"run_practice: env={env_name} model={model_name} -> {ctx.obj['paths'].root}",
             parameters=loaded_parameters)


def _pool(obj) -> EnvPool:
    if "pool" not in obj:
        obj["pool"] = EnvPool(spec=obj["spec"], parameters=obj["parameters"])
    return obj["pool"]


def _close(obj) -> None:
    if "pool" in obj:
        obj["pool"].close()


def do_propose(obj, n_tasks):
    from cusi.practice.stages.propose import propose
    propose(spec=obj["spec"], pool=_pool(obj), vlm=obj["vlm"], paths=obj["paths"], n_tasks=n_tasks,
            max_new_tokens=obj["max_new_tokens"], overwrite=obj["overwrite"], parameters=obj["parameters"])


def do_attempt(obj, max_attempts, lookback):
    from cusi.practice.stages.attempt import attempt
    attempt(spec=obj["spec"], pool=_pool(obj), vlm=obj["vlm"], paths=obj["paths"], max_attempts=max_attempts,
            lookback=lookback, judge_max_new_tokens=obj["max_new_tokens"], executor_kwargs=obj["executor_kwargs"],
            overwrite=obj["overwrite"], parameters=obj["parameters"])


def do_guidance(obj, max_obs_at_once):
    from cusi.practice.stages.guidance import guidance
    guidance(spec=obj["spec"], vlm=obj["vlm"], paths=obj["paths"], max_new_tokens=obj["max_new_tokens"],
             max_obs_at_once=max_obs_at_once, overwrite=obj["overwrite"], parameters=obj["parameters"])


def do_practice(obj, n_attempts, n_random_actions, lookback):
    from cusi.practice.stages.practice import practice
    practice(spec=obj["spec"], pool=_pool(obj), vlm=obj["vlm"], paths=obj["paths"], n_attempts=n_attempts,
             n_random_actions=n_random_actions, lookback=lookback, judge_max_new_tokens=obj["max_new_tokens"],
             executor_kwargs=obj["executor_kwargs"], overwrite=obj["overwrite"], parameters=obj["parameters"])


def do_clean(obj, k, safety_margin):
    from cusi.practice.stages.clean import clean
    clean(spec=obj["spec"], vlm=obj["vlm"], paths=obj["paths"], k=k, safety_margin=safety_margin,
          max_new_tokens=obj["max_new_tokens"], overwrite=obj["overwrite"], parameters=obj["parameters"])


def do_dataset(obj, safety_margin, val_frac, seed):
    from cusi.practice.stages.dataset import build_dataset
    build_dataset(spec=obj["spec"], paths=obj["paths"], safety_margin=safety_margin, val_frac=val_frac, seed=seed,
                  overwrite=obj["overwrite"], parameters=obj["parameters"])


n_tasks_opt = click.option("--n_tasks", default=3, help="Proposed tasks kept per scene.")
max_attempts_opt = click.option("--max_attempts", default=5, help="Tries per task (critique hint between tries).")
lookback_opt = click.option("--lookback", default=8, help="Final frames shown to the judge.")
obs_opt = click.option("--max_obs_at_once", default=8, help="Frames per guidance slice.")
n_attempts_opt = click.option("--n_attempts", default=3, help="Practice episodes per successful task.")
n_random_opt = click.option("--n_random_actions", default=5, help="Random actions perturbing each practice start.")
k_opt = click.option("--k", default=3, help="Paraphrases per task.")
margin_opt = click.option("--safety_margin", default=2, help="Calls kept past the safe-success cutoff.")
val_opt = click.option("--val_frac", default=0.2)
seed_opt = click.option("--seed", default=0, help="Episode split seed.")


@main.command()
@n_tasks_opt
@click.pass_obj
def propose(obj, n_tasks):
    """Propose tasks from each scene's first frame."""
    try:
        do_propose(obj, n_tasks)
    finally:
        _close(obj)


@main.command()
@max_attempts_opt
@lookback_opt
@click.pass_obj
def attempt(obj, max_attempts, lookback):
    """Attempt each proposed task, judged, with critique-hint retries."""
    try:
        do_attempt(obj, max_attempts, lookback)
    finally:
        _close(obj)


@main.command()
@obs_opt
@click.pass_obj
def guidance(obj, max_obs_at_once):
    """Distil guidance from each successful attempt."""
    do_guidance(obj, max_obs_at_once)


@main.command()
@n_attempts_opt
@n_random_opt
@lookback_opt
@click.pass_obj
def practice(obj, n_attempts, n_random_actions, lookback):
    """Practise each successful task from perturbed starts."""
    try:
        do_practice(obj, n_attempts, n_random_actions, lookback)
    finally:
        _close(obj)


@main.command()
@k_opt
@margin_opt
@click.pass_obj
def clean(obj, k, safety_margin):
    """Paraphrase tasks and ACCEPT/REJECT each candidate call."""
    do_clean(obj, k, safety_margin)


@main.command()
@margin_opt
@val_opt
@seed_opt
@click.pass_obj
def dataset(obj, safety_margin, val_frac, seed):
    """Build train/validation JSONL from the cleaned practice data."""
    do_dataset(obj, safety_margin, val_frac, seed)


@main.command(name="all")
@n_tasks_opt
@max_attempts_opt
@lookback_opt
@obs_opt
@n_attempts_opt
@n_random_opt
@k_opt
@margin_opt
@val_opt
@seed_opt
@click.pass_obj
def run_all(obj, n_tasks, max_attempts, lookback, max_obs_at_once, n_attempts, n_random_actions, k, safety_margin,
            val_frac, seed):
    """All six stages in order."""
    try:
        if obj["source"] == "zeroshot":
            do_propose(obj, n_tasks)
            do_attempt(obj, max_attempts, lookback)
        do_guidance(obj, max_obs_at_once)
        do_practice(obj, n_attempts, n_random_actions, lookback)
        do_clean(obj, k, safety_margin)
        do_dataset(obj, safety_margin, val_frac, seed)
    finally:
        _close(obj)


if __name__ == "__main__":
    main()
