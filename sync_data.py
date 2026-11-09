"""
Syncs named sets of storage_dir artifacts with a Hugging Face dataset repo whose layout mirrors storage_dir.

    evals        eval/<env>/<run>[_s<i>]            an eval run (--run_name required)
    trees        search/<env>/<run>                 a search tree with its annotations, task_coverage and task_diversity
                                                    (viewer/ is regenerable and skipped unless --with_viewer)
    practice     practice/<env>/<model>/tree/<run>  practice on the tree <run>, for every model
    practice-ft  models/finetune/<env>/<run>        LoRA fine-tunes on <run>'s practice (checkpoint/ skipped unless
                                                    --with_checkpoints)
For trees, practice and practice-ft, --run_name is comma-separated tree names, or every run when omitted.

Only finished runs are pushed: a run still being written (see each set's blockers, plus a quiet period with no file
modified) is skipped, or is an error when named in --run_name. The Hub limits files per repo and per folder, so a
set's many-file directories go up as uncompressed tars (<name>.tar of the top-level dirs matching a pattern, members
named from the run directory; rebuilt on every push but deterministically, so unchanged dirs upload nothing new),
unpacked into the run directory on pull.
Both push and pull are dry runs unless given --no_dry_run.

    python sync_data.py push --set trees --envs gameboy --run_name sat_random_s0
    python sync_data.py pull --set practice-ft --run_name sat_random_s0
"""
import fnmatch
import functools
import glob
import json
import os
import shlex
import shutil
import sys
import tarfile
import time
from typing import Callable, NamedTuple
import click
from huggingface_hub import CommitOperationAdd, HfApi
from cusi.utils import log_error, log_info, log_warn
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import eval_run_dirs, practice_root, search_run, web_task_file

loaded_parameters = load_parameters()

ENVS = ("gameboy", "web", "android")
#: ROMs are copyrighted; never upload one, whatever a set's dirs contain.
ROM_PATTERNS = ["*.gb", "*.gbc", "*.gba", "*.sav"]
PACK_SUFFIX = ".tar"


def _rel(*, parameters: dict, path: str) -> str:
    return os.path.relpath(path, parameters["storage_dir"])


def _runs(run_name) -> list:
    return [r.strip() for r in run_name.split(",") if r.strip()] if run_name else ["*"]


def _existing(pattern: str) -> list:
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def _newer(path: str, than: str) -> bool:
    return os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(than)


# --------------------------------------------------------------------------- evals

def _eval_inputs(*, parameters: dict) -> list:
    """Shared inputs an eval run needs to be resumed or judged elsewhere."""
    return [web_task_file(parameters=parameters)]


def evals_paths(*, parameters: dict, run_name, envs: list) -> list:
    if not run_name:
        log_error("--set evals needs --run_name.", parameters=parameters)
    return [d for env in envs for d in eval_run_dirs(parameters=parameters, env=env, run=run_name)]


def evals_extra(*, parameters: dict) -> list:
    return [p for p in _eval_inputs(parameters=parameters) if os.path.exists(p)]


def evals_pull(*, parameters: dict, run_name, envs: list) -> list:
    if not run_name:
        log_error("--set evals needs --run_name.", parameters=parameters)
    patterns = [f"eval/{env}/{run_name}/*" for env in envs]
    if "android" in envs:
        patterns.append(f"eval/android/{run_name}_s[0-9]*/*")
    for path in _eval_inputs(parameters=parameters):
        rel = _rel(parameters=parameters, path=path)
        patterns.append(rel if os.path.splitext(rel)[1] else f"{rel}/*")
    return patterns


@functools.lru_cache(maxsize=None)
def _gameboy_task_count(*, game: str, n_tasks) -> int:
    from cusi.eval.gameboy import select_tasks
    return len(select_tasks(game=game, n_tasks=n_tasks))


def _expected_tasks(*, parameters: dict, config: dict) -> int:
    """How many results.jsonl rows a complete run of config has (its own task list, else the env's full set)."""
    tasks = config.get("tasks")
    if isinstance(tasks, str):
        tasks = [t for t in tasks.split(",") if t.strip()]
    if config["env"] == "gameboy":
        return len(tasks) if tasks else _gameboy_task_count(game=config["game"], n_tasks=config.get("n_tasks"))
    if config["env"] == "web":
        if tasks:
            return len(tasks)
        with open(os.path.join(parameters["storage_dir"], config["task_file"])) as f:
            return sum(1 for line in f if line.strip())
    from cusi.eval.android import shard_of, suite_tasks
    names = suite_tasks(tasks=",".join(tasks) if tasks else None)
    if config.get("shard"):
        names = shard_of(names=names, shard=config["shard"])
    return len(names) * config.get("n_task_combinations", 1)


def evals_blockers(*, parameters: dict, path: str) -> list:
    """An eval is finished once its last launch summarised every task of its config, each run to a score, and (Android)
    every shard of the run exists. Android's base <run> dir holds only timing and logs, and no config."""
    config_path, summary_path = os.path.join(path, "config.json"), os.path.join(path, "summary.json")
    if not os.path.exists(config_path):
        return []
    if not os.path.exists(summary_path):
        return ["eval running or killed (no summary.json)"]
    if _newer(os.path.join(path, "results.jsonl"), summary_path):
        return ["eval resumed after its summary.json"]
    with open(config_path) as f:
        config = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    out = []
    expected = _expected_tasks(parameters=parameters, config=config)
    if summary["n"] < expected:
        out.append(f"{summary['n']}/{expected} tasks")
    if summary["n_failed_to_run"]:
        out.append(f"{summary['n_failed_to_run']} task(s) failed to run")
    if config.get("shard"):
        base, n = path.rsplit("_s", 1)[0], int(config["shard"].split("/")[1])
        missing = [i for i in range(n) if not os.path.isdir(f"{base}_s{i}")]
        if missing:
            out.append(f"shards {missing} of {n} missing")
    return out


# --------------------------------------------------------------------------- trees

def trees_paths(*, parameters: dict, run_name, envs: list) -> list:
    return [d for env in envs for r in _runs(run_name)
            for d in _existing(search_run(parameters=parameters, env=env, run=r))]


def trees_pull(*, parameters: dict, run_name, envs: list) -> list:
    return [f"{_rel(parameters=parameters, path=search_run(parameters=parameters, env=env, run=r))}/*"
            for env in envs for r in _runs(run_name)]


def trees_blockers(*, parameters: dict, path: str) -> list:
    """A tree is finished once its search has stopped (and not resumed since), and any annotation, task_coverage or
    task_diversity started on it has finished. An unannotated tree is finished."""
    from cusi.search.annotate import ANNOTATIONS_FILE, unannotated
    from cusi.search.store import EXPANSIONS_FILE, node_rows
    out = []
    summary = os.path.join(path, "summary.json")
    if not os.path.exists(summary):
        if os.path.basename(path) not in parameters.get("sync_time_limited", ()):
            out.append("search running or killed (no summary.json; --time_limited if it ended at its time limit)")
    elif _newer(os.path.join(path, EXPANSIONS_FILE), summary):
        out.append("search resumed after its summary.json")
    if os.path.exists(os.path.join(path, ANNOTATIONS_FILE)):
        nodes = node_rows(directory=path)
        pending = len(unannotated(nodes=nodes))
        if pending:
            out.append(f"annotation unfinished ({pending}/{len(nodes)} nodes left)")
    for stage in ("task_coverage", "task_diversity"):
        for embed_dir in _existing(os.path.join(path, stage, "*")):
            if not os.path.exists(os.path.join(embed_dir, "config.json")):
                out.append(f"{os.path.relpath(embed_dir, path)} embed unfinished")
            for judge_dir in _existing(os.path.join(embed_dir, "judge_*")):
                done = os.path.join(judge_dir, "summary.json")   # written last
                if not os.path.exists(done) or any(_newer(f, done) for f in glob.glob(os.path.join(judge_dir, "*"))):
                    out.append(f"{os.path.relpath(judge_dir, path)} unfinished")
    return out


# --------------------------------------------------------------------------- practice

def _practice_dir(*, parameters: dict, env: str, run: str) -> str:
    """practice_root for every model: practice/<env>/*/tree/<run>."""
    return practice_root(parameters=parameters, env=env, model_name="*", source=run)


def practice_paths(*, parameters: dict, run_name, envs: list) -> list:
    return [d for env in envs for r in _runs(run_name)
            for d in _existing(_practice_dir(parameters=parameters, env=env, run=r))]


def practice_pull(*, parameters: dict, run_name, envs: list) -> list:
    return [f"{_rel(parameters=parameters, path=_practice_dir(parameters=parameters, env=env, run=r))}/*"
            for env in envs for r in _runs(run_name)]


def practice_blockers(*, parameters: dict, path: str) -> list:
    """Finished once it has a dataset/, or (test-set practice, attempt_test: tries only) once its results.csv,
    written at the end of every attempt_test run, is newer than every episode."""
    if os.path.isdir(os.path.join(path, "dataset")):
        return []
    results, episodes = os.path.join(path, "results.csv"), os.path.join(path, "episodes")
    if os.path.exists(results) and os.path.isdir(episodes):
        if any(_newer(os.path.join(episodes, n), results) for n in os.listdir(episodes)):
            return ["attempt_test running (an episode is newer than results.csv)"]
        return []
    return ["no dataset/: practice is unfinished"]


# --------------------------------------------------------------------------- practice-ft

def _finetune_source(*, parameters: dict, env: str, run: str) -> str:
    """Every fine-tune on run's practice (finetune_run without its last part)."""
    return os.path.join(parameters["model_dir"], "finetune", env, run)


def practice_ft_paths(*, parameters: dict, run_name, envs: list) -> list:
    return [d for env in envs for r in _runs(run_name)
            for d in _existing(_finetune_source(parameters=parameters, env=env, run=r))]


def practice_ft_pull(*, parameters: dict, run_name, envs: list) -> list:
    return [f"{_rel(parameters=parameters, path=_finetune_source(parameters=parameters, env=env, run=r))}/*"
            for env in envs for r in _runs(run_name)]


def practice_ft_blockers(*, parameters: dict, path: str) -> list:
    return [f"{os.path.basename(d)} has no adapter/: the fine-tune is unfinished"
            for d in _existing(os.path.join(path, "*")) if not os.path.isdir(os.path.join(d, "adapter"))]


# --------------------------------------------------------------------------- sets

class FileSet(NamedTuple):
    paths: Callable      # (parameters, run_name, envs) -> run directories to push
    pull: Callable       # (parameters, run_name, envs) -> repo patterns to pull
    blockers: Callable   # (parameters, run directory) -> why it is not finished yet ([] when finished)
    packed: dict = {}    # <name>.tar -> the pattern of the run directory's top-level dirs it holds
    extra: Callable = lambda *, parameters: []   # loose files pushed alongside, ungated


def _dirs(*names) -> dict:
    return {n: n for n in names}


SETS = {"evals": FileSet(paths=evals_paths, pull=evals_pull, blockers=evals_blockers, extra=evals_extra,
                         packed={**_dirs("episodes", "trajectories"), "webvoyager_tasks": "task*"}),   # web_task_dir
        "trees": FileSet(paths=trees_paths, pull=trees_pull, blockers=trees_blockers,
                         packed=_dirs("frames", "embeddings", "states", "policies", "curiosity")),   # WRITE_ONCE_DIRS
        "practice": FileSet(paths=practice_paths, pull=practice_pull, blockers=practice_blockers,
                            packed=_dirs("episodes", "dataset")),
        "practice-ft": FileSet(paths=practice_ft_paths, pull=practice_ft_pull, blockers=practice_ft_blockers)}


def _skip_patterns(*, with_viewer: bool, with_checkpoints: bool) -> list:
    """ROMs, plus the large regenerable or intermediate dirs not asked for, at any depth."""
    dirs = [n for n, keep in (("viewer", with_viewer), ("checkpoint", with_checkpoints)) if not keep]
    return ROM_PATTERNS + [p for n in dirs for p in (f"{n}/*", f"*/{n}/*")]


class Scan(NamedTuple):
    loose: list      # files pushed as they are
    packs: dict      # pack name -> its files
    newest: float    # latest mtime of any file in the directory, skipped ones included


def _scan(*, path: str, skip: list, packed: dict) -> Scan:
    loose, packs, newest = [], {}, 0.0
    for root, _, names in os.walk(path):
        for n in names:
            full = os.path.join(root, n)
            newest = max(newest, os.path.getmtime(full))
            rel = os.path.relpath(full, path)
            if any(fnmatch.fnmatch(rel, p) for p in skip):
                continue
            top = rel.split(os.sep)[0]
            pack = next((name for name, pattern in packed.items() if fnmatch.fnmatch(top, pattern)), None)
            if pack is not None and top != rel:
                packs.setdefault(pack, []).append(full)
            else:
                loose.append(full)
    return Scan(loose=loose, packs=packs, newest=newest)


def _size(files: list) -> int:
    return sum(os.path.getsize(f) for f in files)


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def _pack(*, src_dir: str, files: list, out: str) -> None:
    """An uncompressed tar of files (named relative to src_dir) with fixed metadata and order, so the same files give
    the same bytes."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tarfile.open(out, "w", format=tarfile.GNU_FORMAT) as tar:
        for full in sorted(files):
            info = tarfile.TarInfo(os.path.relpath(full, src_dir))
            info.size, info.mode = os.path.getsize(full), 0o644
            with open(full, "rb") as f:
                tar.addfile(info, f)


def _unpack(*, tar_path: str) -> str:
    """Extract a pack into its run directory (the tar's own), then delete the tar."""
    dest = os.path.dirname(tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")
    os.remove(tar_path)
    return dest


def _finished(*, parameters: dict, file_set: str, paths: list, explicit: bool, quiet_minutes: float,
              skip: list) -> list:
    """(path, scan) of each finished path. Unfinished ones are skipped with their reasons (an error when named)."""
    spec, out = SETS[file_set], []
    for path in paths:
        reasons, scan = spec.blockers(parameters=parameters, path=path), None
        if not reasons:
            scan = _scan(path=path, skip=skip, packed=spec.packed)
            idle = (time.time() - scan.newest) / 60
            if idle < quiet_minutes:
                reasons = [f"a file changed {idle:.0f} min ago (< --quiet_minutes {quiet_minutes:g})"]
        if reasons:
            msg = f"[{file_set}] {_rel(parameters=parameters, path=path)} is not finished: {'; '.join(reasons)}"
            if explicit:
                log_error(msg, parameters=parameters)
            log_warn(f"{msg}. Skipped.", parameters=parameters)
        else:
            out.append((path, scan))
    return out


# --------------------------------------------------------------------------- commands

def _envs(envs: str) -> list:
    out = [e.strip() for e in envs.split(",") if e.strip()]
    bad = [e for e in out if e not in ENVS]
    if bad:
        raise click.UsageError(f"Unknown envs {bad}; choose from {list(ENVS)}.")
    return out


def no_dry_run_command() -> str:
    argv = [a for a in sys.argv[1:] if a != "--dry_run"]
    return " ".join(shlex.quote(p) for p in ["python", os.path.basename(sys.argv[0])] + argv + ["--no_dry_run"])


def selection_options(command):
    command = click.option("--dry_run/--no_dry_run", default=True, show_default=True,
                           help="--dry_run only reports what would be sent or written.")(command)
    command = click.option("--with_checkpoints", is_flag=True,
                           help="practice-ft: also sync the trainer's checkpoint/ dirs.")(command)
    command = click.option("--with_viewer", is_flag=True, help="trees: also sync the regenerable viewer/.")(command)
    command = click.option("--envs", default=",".join(ENVS), show_default=True, help="Comma-separated envs.")(command)
    command = click.option("--run_name", default=None,
                           help="evals: the eval run. Other sets: comma-separated tree names (default: all).")(command)
    return click.option("--set", "file_set", required=True, type=click.Choice(sorted(SETS)))(command)


@click.command()
@click.option("--private/--public", default=True, show_default=True)
@click.pass_obj
def create_hub_repo(parameters, private):
    """Create the dataset repo on the Hub (no-op if it exists)."""
    parameters["api"].create_repo(repo_id=parameters["repo_id"], repo_type="dataset", exist_ok=True, private=private)
    log_info(f"Repo {parameters['repo_id']} ready.", parameters=parameters)


@click.command()
@selection_options
@click.option("--quiet_minutes", default=30.0, show_default=True,
              help="A run with any file modified this recently is still being written, and is not pushed.")
@click.option("--time_limited", default=None,
              help="trees: comma-separated tree names whose search ended at its Slurm time limit (chained segments, "
                   "no summary.json); they are finished once quiet.")
@click.pass_obj
def push(parameters, file_set, run_name, envs, with_viewer, with_checkpoints, dry_run, quiet_minutes, time_limited):
    """Upload a set's finished runs, one commit each; the repo path of each file is its path under storage_dir."""
    parameters["sync_time_limited"] = tuple(_runs(time_limited)) if time_limited else ()
    api, repo_id, spec = parameters["api"], parameters["repo_id"], SETS[file_set]
    skip = _skip_patterns(with_viewer=with_viewer, with_checkpoints=with_checkpoints)
    paths = spec.paths(parameters=parameters, run_name=run_name, envs=_envs(envs))
    if not paths:
        log_error(f"Nothing to push for --set {file_set} --run_name {run_name} in {envs}.", parameters=parameters)
    ready = _finished(parameters=parameters, file_set=file_set, paths=paths, explicit=run_name is not None,
                      quiet_minutes=quiet_minutes, skip=skip)
    extra = spec.extra(parameters=parameters)
    hub_files = total = 0
    for path, scan in ready:
        rel = _rel(parameters=parameters, path=path)
        size = _size(scan.loose) + sum(_size(f) for f in scan.packs.values())
        hub_files, total = hub_files + len(scan.loose) + len(scan.packs), total + size
        packs = "".join(f", {d}{PACK_SUFFIX} ({len(f)} files)" for d, f in sorted(scan.packs.items()))
        log_info(f"[{file_set}] {rel}: {len(scan.loose)} loose file(s){packs}; {_gb(size)}", parameters=parameters)
        if dry_run:
            continue
        stage = os.path.join(parameters["tmp_dir"], "sync", rel)
        ops = [CommitOperationAdd(path_in_repo=_rel(parameters=parameters, path=f), path_or_fileobj=f)
               for f in scan.loose]
        try:
            for d, files in sorted(scan.packs.items()):
                tar_path = os.path.join(stage, d + PACK_SUFFIX)
                _pack(src_dir=path, files=files, out=tar_path)
                ops.append(CommitOperationAdd(path_in_repo=f"{rel}/{d}{PACK_SUFFIX}", path_or_fileobj=tar_path))
            api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops, commit_message=f"{file_set}: {rel}")
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        log_info(f"Pushed {rel} -> {repo_id}", parameters=parameters)
    for f in extra:
        rel = _rel(parameters=parameters, path=f)
        log_info(f"[{file_set}] {rel}: 1 file", parameters=parameters)
        if not dry_run:
            api.upload_file(repo_id=repo_id, repo_type="dataset", path_or_fileobj=f, path_in_repo=rel,
                            commit_message=f"{file_set}: {rel}")
    log_info(f"[{file_set}] {len(ready)}/{len(paths)} run(s) finished: {hub_files + len(extra)} Hub file(s), "
             f"{_gb(total)} -> {repo_id} (skipping {skip})", parameters=parameters)
    if dry_run:
        log_info(f"Dry run: nothing sent. To push, run:\n    {no_dry_run_command()}", parameters=parameters)


def _is_pack(*, filename: str, packed: dict) -> bool:
    name = os.path.basename(filename)
    return name.endswith(PACK_SUFFIX) and name[:-len(PACK_SUFFIX)] in packed


@click.command()
@selection_options
@click.pass_obj
def pull(parameters, file_set, run_name, envs, with_viewer, with_checkpoints, dry_run):
    """Download a set into storage_dir, each file back at its own path and each pack unpacked into its run dir."""
    api, repo_id, local_dir, spec = parameters["api"], parameters["repo_id"], parameters["storage_dir"], SETS[file_set]
    patterns = spec.pull(parameters=parameters, run_name=run_name, envs=_envs(envs))
    kwargs = dict(repo_id=repo_id, repo_type="dataset", local_dir=local_dir, allow_patterns=patterns,
                  ignore_patterns=_skip_patterns(with_viewer=with_viewer, with_checkpoints=with_checkpoints))
    result = api.snapshot_download(**kwargs, dry_run=True)
    # A tar is deleted once unpacked, so it always downloads again; what it overwrites is the dirs it holds.
    packs = [f for f in result if _is_pack(filename=f.filename, packed=spec.packed)]
    loose = [f for f in result if not _is_pack(filename=f.filename, packed=spec.packed)]
    incoming = [f for f in loose if f.will_download]
    overwrite = [f.local_path for f in incoming if os.path.exists(f.local_path)]
    for f in packs:
        pattern = spec.packed[os.path.basename(f.filename)[:-len(PACK_SUFFIX)]]
        overwrite += [f"{d}/" for d in _existing(os.path.join(os.path.dirname(f.local_path), pattern))]
    log_info(f"[{file_set}] {len(result)} file(s) matched in {repo_id}: {len(packs)} pack(s), "
             f"{len(incoming)} loose file(s) to download, {len(loose) - len(incoming)} up to date; "
             f"{len(overwrite)} local path(s) would be overwritten -> {local_dir}", parameters=parameters)
    for p in overwrite:
        log_warn(f"would overwrite {p}", parameters=parameters)
    if dry_run:
        log_info(f"Dry run: nothing written. To pull, run:\n    {no_dry_run_command()}", parameters=parameters)
        return
    api.snapshot_download(**kwargs)
    for f in packs:
        _unpack(tar_path=f.local_path)
        log_info(f"Unpacked {f.filename}", parameters=parameters)
    log_info(f"Pulled {len(incoming)} file(s) and {len(packs)} pack(s) into {local_dir}", parameters=parameters)


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
