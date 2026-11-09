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


def do_attempt(obj, max_tries, lookback, max_tasks, sample_tasks, sample_seed):
    from cusi.practice.stages.attempt import attempt
    from cusi.search.restore import NodeRestorer
    restorer = NodeRestorer(directory=search_run(parameters=obj["parameters"], env=obj["env_name"],
                                                 run=obj["source"]), parameters=obj["parameters"],
                            state_cache=obj["state_cache"])
    attempt(spec=obj["spec"], pool=_pool(obj), vlm=obj["vlm"], paths=obj["paths"], restorer=restorer,
            max_tries=max_tries, lookback=lookback, judge_max_new_tokens=obj["max_new_tokens"], max_tasks=max_tasks,
            sample_tasks=sample_tasks, sample_seed=sample_seed, parameters=obj["parameters"])


def do_dataset(obj, n_paraphrases, safety_margin, val_frac, seed, labels="judge"):
    from cusi.practice.stages.dataset import build_dataset
    build_dataset(spec=obj["spec"], vlm=obj["vlm"], paths=obj["paths"], n_paraphrases=n_paraphrases,
                  safety_margin=safety_margin, val_frac=val_frac, seed=seed, max_new_tokens=obj["max_new_tokens"],
                  overwrite=obj["overwrite"], parameters=obj["parameters"], labels=labels)


max_tries_opt = click.option("--max_tries", default=3, help="Tries per task (a critique hint between tries).")
lookback_opt = click.option("--lookback", default=8, help="Final frames shown to the judge.")
max_tasks_opt = click.option("--max_tasks", default=None, type=int, help="Attempt only the first this many tasks.")
sample_opt = click.option("--sample_tasks", default=None, type=int,
                          help="Attempt the first this many tasks of a seeded shuffle (default: all); raise it "
                               "to extend the sample.")
sample_seed_opt = click.option("--sample_seed", default=0, show_default=True, help="Seed of --sample_tasks.")
paraphrases_opt = click.option("--n_paraphrases", default=3, help="Paraphrases per successful train task.")
margin_opt = click.option("--safety_margin", default=2, help="Executor calls kept past the safe success point.")
val_opt = click.option("--val_frac", default=0.2, help="Share of tasks held out for validation.")
seed_opt = click.option("--seed", default=0, help="Train/validation task split seed.")


@main.command()
@max_tries_opt
@lookback_opt
@max_tasks_opt
@sample_opt
@sample_seed_opt
@click.pass_obj
def attempt(obj, max_tries, lookback, max_tasks, sample_tasks, sample_seed):
    """Attempt each task, judged, with critique-hint retries; one pickle per task."""
    try:
        do_attempt(obj, max_tries, lookback, max_tasks, sample_tasks, sample_seed)
    finally:
        _close(obj)


@main.command()
@paraphrases_opt
@margin_opt
@val_opt
@seed_opt
@click.option("--labels", default="judge", type=click.Choice(["judge", "gt"]),
              help="Label tries by the judge, or by the benchmark's success (test-set practice only).")
@click.pass_obj
def dataset(obj, n_paraphrases, safety_margin, val_frac, seed, labels):
    """Build the success and failure train/validation JSONL from the attempts."""
    do_dataset(obj, n_paraphrases, safety_margin, val_frac, seed, labels=labels)


@main.command(name="attempt_test")
@click.option("--repeats", required=True, help="Repeats to attempt, comma-separated (e.g. 0,1).")
@click.option("--game", required=True, help="The GameBoyWorlds game whose benchmark tasks are practised.")
@click.option("--n_workers", default=8, show_default=True, help="Tasks attempted in parallel.")
@max_tries_opt
@lookback_opt
@click.pass_obj
def attempt_test(obj, repeats, game, n_workers, max_tries, lookback):
    """Practice on the test set (GameBoy, a diagnostic): attempt each benchmark task from its start state as the
    eval runs it, judged with critique-hint retries, recording the benchmark's success beside the judge's."""
    from cusi.practice.stages.attempt_test import attempt_test as run
    run(spec=obj["spec"], vlm=obj["vlm"], paths=obj["paths"], game=game,
        repeats=[int(r) for r in repeats.split(",")], max_tries=max_tries, lookback=lookback,
        judge_max_new_tokens=obj["max_new_tokens"], n_workers=n_workers, parameters=obj["parameters"])


@main.command(name="focus_dataset")
@click.option("--from_source", required=True, help="The attempt_test run whose tries are the focus rows.")
@click.option("--tasks", required=True, help="Benchmark row indices to focus on, comma-separated.")
@click.option("--labels", required=True, type=click.Choice(["judge", "gt"]), help="Which tries count as successes.")
@click.option("--mix_source", required=True, help="A practice run of this env whose dataset is mixed in and "
                                                   "validated on.")
@click.option("--mix_ratio", default=1.0, show_default=True, help="Mix-in rows per focus row (0: none).")
@click.option("--focus_repeat", default=1, show_default=True, help="Copies of each focus row per epoch.")
@click.option("--val_rows", default=400, show_default=True, help="Validation rows sampled from the mix dataset.")
@margin_opt
@seed_opt
@click.pass_obj
def focus_dataset(obj, from_source, tasks, labels, mix_source, mix_ratio, focus_repeat, val_rows, safety_margin,
                  seed):
    """A focus dataset: a few test tasks' practice successes, mixed with another practice dataset (GameBoy)."""
    from cusi.practice.stages.focus import build_focus_dataset

    def other(source):
        return PracticePaths(parameters=obj["parameters"], env_name=obj["env_name"], model_name=obj["vlm"].model_name,
                             source=source)
    build_focus_dataset(paths=obj["paths"], from_paths=other(from_source), tasks=[int(t) for t in tasks.split(",")],
                        labels=labels, mix_paths=other(mix_source), mix_ratio=mix_ratio, focus_repeat=focus_repeat,
                        val_rows=val_rows, safety_margin=safety_margin, seed=seed, overwrite=obj["overwrite"],
                        parameters=obj["parameters"])


@main.command(name="all")
@max_tries_opt
@lookback_opt
@max_tasks_opt
@sample_opt
@sample_seed_opt
@paraphrases_opt
@margin_opt
@val_opt
@seed_opt
@click.pass_obj
def run_all(obj, max_tries, lookback, max_tasks, sample_tasks, sample_seed, n_paraphrases, safety_margin, val_frac,
            seed):
    """attempt, dataset."""
    try:
        do_attempt(obj, max_tries, lookback, max_tasks, sample_tasks, sample_seed)
        do_dataset(obj, n_paraphrases, safety_margin, val_frac, seed)
    finally:
        _close(obj)


if __name__ == "__main__":
    main()
