"""
World models trained on search trees (cusi.world_model), one env per process. Each name has its own directory,
storage/world_model/<env>/<name>/{data, embedder, wm}:

    python run_world_model.py --env <env> data --name <name> --runs <tree>,<tree>,...
    python run_world_model.py --env <env> embedder --name <name> --encoder_model <SigLIP 2 id>
    python run_world_model.py --env <env> wm --name <name>
    python run_world_model.py --env <env> all --name <name> --runs ... --encoder_model ...   # the three in order
    python run_world_model.py --env <env> test --name <name> --eval_runs <eval run>,... --tag <tag>   # on real steps
    python run_world_model.py --env <env> seeds_data --study <s> --group gemma=<tree>,... --group random=<tree>,...
    python run_world_model.py --env <env> seeds_wm --study <s> --embedder <name> --eval_runs ... --tag <tag>
    python run_world_model.py --env <env> seeds_compare --study <s> --tag <tag>   # one WM per tree seed, compared
"""
import json
import os
import click
from cusi.utils.parameter_handling import load_parameters, compute_secondary_parameters
from cusi.utils.log_handling import log_info
from cusi.utils.paths import search_run
from cusi.agents.specs import ENV_NAMES
from cusi.world_model import model_dirs

loaded_parameters = load_parameters()


@click.group()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--random_seed", default=loaded_parameters["random_seed"])
@click.pass_context
def main(ctx, env_name, random_seed):
    loaded_parameters["random_seed"] = random_seed
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = {"parameters": loaded_parameters, "env_name": env_name}


def _data(*, obj: dict, name: str, runs: str, train_frac: float, workers: int) -> dict:
    from cusi.world_model.dataset import build_dataset
    p, env = obj["parameters"], obj["env_name"]
    run_dirs = [search_run(parameters=p, env=env, run=r.strip()) for r in runs.split(",") if r.strip()]
    missing = [d for d in run_dirs if not os.path.isdir(d)]
    if missing:
        raise click.UsageError(f"no such trees: {missing}")
    return build_dataset(env_name=env, run_dirs=run_dirs, out_dir=model_dirs(parameters=p, env=env, name=name)["data"],
                         train_frac=train_frac, seed=p["random_seed"], workers=workers, parameters=p)


def _embedder(*, obj: dict, name: str, encoder_model: str, max_steps: int, max_minutes: float, batch_size: int,
              lr: float, decoder_lr: float, eval_every: int, patience: int, max_val_frames: int, workers: int) -> dict:
    from cusi.world_model.dataset import read_dataset
    from cusi.world_model.embedder import train_siglip_early_stopping
    p, env = obj["parameters"], obj["env_name"]
    dirs = model_dirs(parameters=p, env=env, name=name)
    data = read_dataset(directory=dirs["data"])
    return train_siglip_early_stopping(
        env_name=env, train_paths=[r["path"] for r in data["frames_train"]],
        val_paths=[r["path"] for r in data["frames_val"]], out_dir=dirs["embedder"], encoder_model=encoder_model,
        max_steps=max_steps, max_minutes=max_minutes, batch_size=batch_size, lr=lr, decoder_lr=decoder_lr,
        eval_every=eval_every, patience=patience, max_val_frames=max_val_frames, workers=workers,
        seed=p["random_seed"], parameters=p)


def _wm(*, obj: dict, name: str, max_epochs: int, patience: int, batch_size: int, lr: float, hidden_dim: int,
        residual: bool, workers: int) -> dict:
    from cusi.world_model.train import train_world_model
    p, env = obj["parameters"], obj["env_name"]
    dirs = model_dirs(parameters=p, env=env, name=name)
    return train_world_model(env_name=env, dataset_dir=dirs["data"], embedder_dir=dirs["embedder"],
                             out_dir=dirs["wm"], max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                             lr=lr, hidden_dim=hidden_dim, residual=residual, seed=p["random_seed"],
                             embed_batch_size=128, workers=workers, parameters=p)


def data_options(f):
    f = click.option("--runs", required=True, help="Comma-separated search tree names (storage/search/<env>/).")(f)
    f = click.option("--train_frac", default=0.85, show_default=True, help="Fraction of expansions in train.")(f)
    return f


def embedder_options(f):
    f = click.option("--encoder_model", required=True, help="SigLIP 2 model id to fine-tune.")(f)
    f = click.option("--embedder_max_steps", default=50_000, show_default=True)(f)
    f = click.option("--embedder_max_minutes", default=600.0, show_default=True)(f)
    f = click.option("--embedder_batch_size", default=32, show_default=True)(f)
    f = click.option("--embedder_lr", default=1e-5, show_default=True)(f)
    f = click.option("--decoder_lr", default=3e-4, show_default=True)(f)
    f = click.option("--eval_every", default=500, show_default=True, help="Optimizer steps between validations.")(f)
    f = click.option("--embedder_patience", default=6, show_default=True,
                     help="Validations without a new best val loss before stopping.")(f)
    f = click.option("--max_val_frames", default=2048, show_default=True, help="Val screens used for validation.")(f)
    return f


def wm_options(f):
    f = click.option("--max_epochs", default=300, show_default=True)(f)
    f = click.option("--wm_patience", default=15, show_default=True,
                     help="Epochs without a new best val loss before stopping.")(f)
    f = click.option("--wm_batch_size", default=1024, show_default=True)(f)
    f = click.option("--wm_lr", default=1e-3, show_default=True)(f)
    f = click.option("--hidden_dim", default=512, show_default=True)(f)
    f = click.option("--residual", default=True, type=bool, show_default=True,
                     help="Predict emb(t) + a learned change (an untrained model predicts no change).")(f)
    return f


def common(f):
    f = click.option("--name", required=True, help="Model name: storage/world_model/<env>/<name>.")(f)
    f = click.option("--workers", default=8, show_default=True, help="Processes decoding frames.")(f)
    return f


@main.command()
@common
@data_options
@click.pass_obj
def data(obj, name, workers, runs, train_frac):
    """Train / val transitions from the trees (split by expansion, val deduplicated against train)."""
    _data(obj=obj, name=name, runs=runs, train_frac=train_frac, workers=workers)


@main.command()
@common
@embedder_options
@click.pass_obj
def embedder(obj, name, workers, encoder_model, embedder_max_steps, embedder_max_minutes, embedder_batch_size,
             embedder_lr, decoder_lr, eval_every, embedder_patience, max_val_frames):
    """Fine-tune SigLIP (with its decoder) on the dataset's train screens, early-stopped on its val screens."""
    _embedder(obj=obj, name=name, encoder_model=encoder_model, max_steps=embedder_max_steps,
              max_minutes=embedder_max_minutes, batch_size=embedder_batch_size, lr=embedder_lr, decoder_lr=decoder_lr,
              eval_every=eval_every, patience=embedder_patience, max_val_frames=max_val_frames, workers=workers)


@main.command()
@common
@wm_options
@click.pass_obj
def wm(obj, name, workers, max_epochs, wm_patience, wm_batch_size, wm_lr, hidden_dim, residual):
    """Train the world model on the fine-tuned SigLIP's embeddings, early-stopped on val loss."""
    _wm(obj=obj, name=name, max_epochs=max_epochs, patience=wm_patience, batch_size=wm_batch_size, lr=wm_lr,
        hidden_dim=hidden_dim, residual=residual, workers=workers)


@main.command(name="all")
@common
@data_options
@embedder_options
@wm_options
@click.pass_obj
def all_stages(obj, name, workers, runs, train_frac, encoder_model, embedder_max_steps, embedder_max_minutes,
               embedder_batch_size, embedder_lr, decoder_lr, eval_every, embedder_patience, max_val_frames, max_epochs,
               wm_patience, wm_batch_size, wm_lr, hidden_dim, residual):
    """data, embedder and wm in order (a stage whose output exists is not redone)."""
    dirs = model_dirs(parameters=obj["parameters"], env=obj["env_name"], name=name)
    if not os.path.exists(os.path.join(dirs["data"], "dataset.json")):
        _data(obj=obj, name=name, runs=runs, train_frac=train_frac, workers=workers)
    if not os.path.exists(os.path.join(dirs["embedder"], "train_summary.json")):
        _embedder(obj=obj, name=name, encoder_model=encoder_model, max_steps=embedder_max_steps,
                  max_minutes=embedder_max_minutes, batch_size=embedder_batch_size, lr=embedder_lr,
                  decoder_lr=decoder_lr, eval_every=eval_every, patience=embedder_patience,
                  max_val_frames=max_val_frames, workers=workers)
    if not os.path.exists(os.path.join(dirs["wm"], "train_summary.json")):
        _wm(obj=obj, name=name, max_epochs=max_epochs, patience=wm_patience, batch_size=wm_batch_size, lr=wm_lr,
            hidden_dim=hidden_dim, residual=residual, workers=workers)
    log_info(f"world model {name} [{obj['env_name']}] done: {dirs['root']}", parameters=obj["parameters"])


@main.command()
@common
@click.option("--eval_runs", required=True,
              help="Comma-separated eval runs (storage/eval/<env>/<run>, with Android's _s<i> shards) whose "
                   "episodes' steps are the test transitions.")
@click.option("--tag", required=True, help="Output name: storage/world_model/<env>/<name>/test/<tag>.json.")
@click.option("--max_k", default=5, show_default=True, help="Longest open-loop rollout.")
@click.option("--batch_size", default=128, show_default=True)
@click.pass_obj
def test(obj, name, workers, eval_runs, tag, max_k, batch_size):
    """The trained world model on real task-attempt transitions (eval episodes): one step and open-loop rollouts."""
    from cusi.utils.paths import eval_run_dirs
    from cusi.world_model.evaluate import evaluate_on_episodes
    p, env = obj["parameters"], obj["env_name"]
    dirs = model_dirs(parameters=p, env=env, name=name)
    run_dirs = [d for r in eval_runs.split(",") if r.strip() for d in eval_run_dirs(parameters=p, env=env, run=r.strip())]
    if not run_dirs:
        raise click.UsageError(f"no such eval runs: {eval_runs}")
    evaluate_on_episodes(wm_dir=dirs["wm"], run_dirs=run_dirs, out_path=os.path.join(dirs["root"], "test", f"{tag}.json"),
                         max_k=max_k, batch_size=batch_size, workers=workers, parameters=p)



def _group_option(f):
    return click.option("--group", "groups", multiple=True, required=True,
                        help="label=tree,tree,... (one tree per seed); repeat per policy. The first two labels are "
                             "compared (first minus second).")(f)


@main.command(name="seeds_data")
@click.option("--study", required=True, help="Study name: storage/world_model/<env>/seeds/<study>.")
@_group_option
@click.option("--train_frac", default=0.85, show_default=True)
@click.option("--workers", default=8, show_default=True)
@click.pass_obj
def seeds_data(obj, study, groups, train_frac, workers):
    """Seed study: one dataset per tree, and copies cut to the smallest tree's train size."""
    from cusi.world_model.seeds import build_seed_datasets
    p, env = obj["parameters"], obj["env_name"]
    parsed = {}
    for g in groups:
        label, _, trees = g.partition("=")
        parsed[label.strip()] = [t.strip() for t in trees.split(",") if t.strip()]
    run_dirs = {t: search_run(parameters=p, env=env, run=t) for trees in parsed.values() for t in trees}
    missing = [d for d in run_dirs.values() if not os.path.isdir(d)]
    if missing:
        raise click.UsageError(f"no such trees: {missing}")
    build_seed_datasets(env_name=env, study=study, groups=parsed, run_dirs=run_dirs, train_frac=train_frac,
                        seed=p["random_seed"], workers=workers, parameters=p)


@main.command(name="seeds_wm")
@click.option("--study", required=True)
@click.option("--embedder", required=True, help="Trained model name whose fine-tuned SigLIP (and decoder) is used.")
@click.option("--eval_runs", required=True, help="Comma-separated eval runs whose episodes' steps are the test set.")
@click.option("--tag", required=True)
@click.option("--max_k", default=5, show_default=True)
@click.option("--workers", default=8, show_default=True)
@wm_options
@click.pass_obj
def seeds_wm(obj, study, embedder, eval_runs, tag, max_k, workers, max_epochs, wm_patience, wm_batch_size, wm_lr,
             hidden_dim, residual):
    """Seed study: a world model per dataset in one embedder, each tested on the eval runs' transitions."""
    from cusi.utils.paths import eval_run_dirs
    from cusi.world_model.seeds import train_and_test_seed_models
    p, env = obj["parameters"], obj["env_name"]
    run_dirs = [d for r in eval_runs.split(",") if r.strip() for d in eval_run_dirs(parameters=p, env=env, run=r.strip())]
    if not run_dirs:
        raise click.UsageError(f"no such eval runs: {eval_runs}")
    train_and_test_seed_models(env_name=env, study=study, embedder=embedder, eval_run_dirs=run_dirs, tag=tag,
                               max_epochs=max_epochs, patience=wm_patience, batch_size=wm_batch_size, lr=wm_lr,
                               hidden_dim=hidden_dim, residual=residual, seed=p["random_seed"], workers=workers,
                               max_k=max_k, parameters=p)


@main.command(name="seeds_compare")
@click.option("--study", required=True)
@click.option("--tag", required=True)
@click.pass_obj
def seeds_compare(obj, study, tag):
    """Seed study: per-seed test metrics of the first two labels compared (Welch t, Mann-Whitney U, Cohen's d)."""
    from cusi.world_model.seeds import compare_seed_models
    compare_seed_models(env_name=obj["env_name"], study=study, tag=tag, parameters=obj["parameters"])


if __name__ == "__main__":
    main()
