"""
Syncs named sets of storage_dir artifacts with a Hugging Face dataset repo whose layout mirrors storage_dir.

    python sync_data.py push --set evals --run_name <run>
    python sync_data.py pull --set evals --run_name <run>            # dry run
"""
import os
import shlex
import sys
import click
from huggingface_hub import HfApi
from cusi.utils import log_error, log_info, log_warn
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import eval_run_dirs, eval_subsets_dir, web_task_file

loaded_parameters = load_parameters()

ENVS = ("gameboy", "web", "android")
#: ROMs are copyrighted; never upload one, whatever a set's dirs contain.
ROM_PATTERNS = ["*.gb", "*.gbc", "*.gba", "*.sav"]


def _rel(*, parameters: dict, path: str) -> str:
    return os.path.relpath(path, parameters["storage_dir"])


def _eval_inputs(*, parameters: dict) -> list:
    """Shared inputs an eval run needs to be resumed or judged elsewhere."""
    return [web_task_file(parameters=parameters), eval_subsets_dir(parameters=parameters, name="mem10")]


def evals_push(*, parameters: dict, run_name: str, envs: list) -> list:
    if not run_name:
        log_error("--set evals needs --run_name.", parameters=parameters)
    dirs = [d for env in envs for d in eval_run_dirs(parameters=parameters, env=env, run=run_name)]
    if not dirs:
        log_error(f"No eval run {run_name!r} for {envs}.", parameters=parameters)
    for d in dirs:     # Android's base <run> dir holds only timing and logs, and no config
        if os.path.exists(os.path.join(d, "config.json")) and not os.path.exists(os.path.join(d, "summary.json")):
            log_warn(f"{_rel(parameters=parameters, path=d)} has no summary.json: the run may be unfinished.",
                     parameters=parameters)
    return dirs + [p for p in _eval_inputs(parameters=parameters) if os.path.exists(p)]


def evals_pull(*, parameters: dict, run_name: str, envs: list) -> list:
    if not run_name:
        log_error("--set evals needs --run_name.", parameters=parameters)
    patterns = [f"eval/{env}/{run_name}/*" for env in envs]
    if "android" in envs:
        patterns.append(f"eval/android/{run_name}_s[0-9]*/*")
    for path in _eval_inputs(parameters=parameters):
        rel = _rel(parameters=parameters, path=path)
        patterns.append(rel if os.path.splitext(rel)[1] else f"{rel}/*")
    return patterns


#: set -> (paths to push, patterns to pull), each a function of the CLI selection.
SETS = {"evals": (evals_push, evals_pull)}


def selection_options(command):
    command = click.option("--envs", default=",".join(ENVS), show_default=True,
                           help="Comma-separated envs (evals).")(command)
    command = click.option("--run_name", default=None, help="Run name (evals).")(command)
    return click.option("--set", "file_set", required=True, type=click.Choice(sorted(SETS)))(command)


def _envs(envs: str) -> list:
    out = [e.strip() for e in envs.split(",") if e.strip()]
    bad = [e for e in out if e not in ENVS]
    if bad:
        raise click.UsageError(f"Unknown envs {bad}; choose from {list(ENVS)}.")
    return out


def no_dry_run_command() -> str:
    argv = [a for a in sys.argv[1:] if a != "--dry_run"]
    return " ".join(shlex.quote(p) for p in ["python", os.path.basename(sys.argv[0])] + argv + ["--no_dry_run"])


@click.command()
@click.option("--private/--public", default=True, show_default=True)
@click.pass_obj
def create_hub_repo(parameters, private):
    """Create the dataset repo on the Hub (no-op if it exists)."""
    parameters["api"].create_repo(repo_id=parameters["repo_id"], repo_type="dataset", exist_ok=True, private=private)
    log_info(f"Repo {parameters['repo_id']} ready.", parameters=parameters)


@click.command()
@selection_options
@click.pass_obj
def push(parameters, file_set, run_name, envs):
    """Upload a set; the repo path of each file is its path under storage_dir. Unchanged files are skipped."""
    api, repo_id = parameters["api"], parameters["repo_id"]
    paths = SETS[file_set][0](parameters=parameters, run_name=run_name, envs=_envs(envs))
    for path in paths:
        rel = _rel(parameters=parameters, path=path)
        if os.path.isdir(path):
            api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=path, path_in_repo=rel,
                              ignore_patterns=ROM_PATTERNS, commit_message=f"{file_set}: {rel}")
        else:
            api.upload_file(repo_id=repo_id, repo_type="dataset", path_or_fileobj=path, path_in_repo=rel,
                            commit_message=f"{file_set}: {rel}")
        log_info(f"Pushed {rel} -> {repo_id}", parameters=parameters)


@click.command()
@selection_options
@click.option("--dry_run/--no_dry_run", default=True, show_default=True,
              help="--dry_run only reports what would be written.")
@click.pass_obj
def pull(parameters, file_set, run_name, envs, dry_run):
    """Download a set into storage_dir, each file back at its own path."""
    api, repo_id, local_dir = parameters["api"], parameters["repo_id"], parameters["storage_dir"]
    patterns = SETS[file_set][1](parameters=parameters, run_name=run_name, envs=_envs(envs))
    kwargs = dict(repo_id=repo_id, repo_type="dataset", local_dir=local_dir, allow_patterns=patterns,
                  ignore_patterns=ROM_PATTERNS)
    result = api.snapshot_download(**kwargs, dry_run=True)
    incoming = [f for f in result if f.will_download]
    overwrite = [f for f in incoming if f.local_path is not None and os.path.exists(f.local_path)]
    log_info(f"[{file_set}] {len(result)} file(s) matched in {repo_id}: {len(incoming) - len(overwrite)} new, "
             f"{len(overwrite)} would be overwritten, {len(result) - len(incoming)} up to date -> {local_dir}",
             parameters=parameters)
    for f in overwrite:
        log_warn(f"would overwrite {f.local_path}", parameters=parameters)
    if dry_run:
        log_info(f"Dry run: nothing written. To pull, run:\n    {no_dry_run_command()}", parameters=parameters)
        return
    api.snapshot_download(**kwargs)
    log_info(f"Pulled {len(incoming)} file(s) into {local_dir}", parameters=parameters)


@click.group()
@click.option("--huggingface_repo_namespace", default=None, help="Overrides the config's.")
@click.option("--huggingface_repo_name", default=None, help="Overrides the config's.")
@click.pass_context
def main(ctx, huggingface_repo_namespace, huggingface_repo_name):
    parameters = loaded_parameters
    namespace = huggingface_repo_namespace or parameters.get("huggingface_repo_namespace")
    name = huggingface_repo_name or parameters.get("huggingface_repo_name")
    if not namespace or not name or "PLACEHOLDER" in (namespace, name):
        log_error("Set huggingface_repo_namespace and huggingface_repo_name in configs/private_vars.yaml "
                  "or pass them as options.", parameters=parameters)
    parameters["repo_id"] = f"{namespace}/{name}"
    parameters["api"] = HfApi()
    ctx.obj = parameters


main.add_command(create_hub_repo, name="init")
main.add_command(push, name="push")
main.add_command(pull, name="pull")

if __name__ == "__main__":
    main()
