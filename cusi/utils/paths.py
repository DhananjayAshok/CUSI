"""
Every shared artifact path under storage_dir, as pure functions of a small identity and the parameters.

    python -m cusi.utils.paths eval_run --env web --run abl_x
"""
import glob
import os
import sys
from cusi.utils.parameter_handling import load_parameters


def _storage(parameters: dict) -> str:
    return parameters["storage_dir"]


def storage_relative(*, parameters: dict, path: str) -> str:
    """A path to store in an artifact: relative to storage_dir when under it, else unchanged."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(_storage(parameters)))
    return path if rel.startswith("..") else rel


def _model_short(model_name: str) -> str:
    """The last part of a model's HF id."""
    return model_name.split("/")[-1]


# --------------------------------------------------------------------------- eval

def eval_root(*, parameters: dict, env: str) -> str:
    return os.path.join(_storage(parameters), "eval", env)


def eval_run(*, parameters: dict, env: str, run: str) -> str:
    return os.path.join(eval_root(parameters=parameters, env=env), run)


def eval_run_dirs(*, parameters: dict, env: str, run: str) -> list:
    """The run's existing directories: <run>, and Android's <run>_s<i> shards."""
    base = eval_run(parameters=parameters, env=env, run=run)
    shards = [d for d in glob.glob(f"{glob.escape(base)}_s*") if d[len(base) + 2:].isdigit()]
    return [d for d in [base] + sorted(shards, key=lambda d: int(d[len(base) + 2:])) if os.path.isdir(d)]


def eval_episode(*, run_dir: str, task_id: str) -> str:
    return os.path.join(run_dir, "episodes", task_id)


def eval_trajectories(*, run_dir: str) -> str:
    return os.path.join(run_dir, "trajectories")


def web_task_dir(*, run_dir: str, task_id: str) -> str:
    """WebVoyager run.py's per-task directory name, which auto_eval expects."""
    return os.path.join(run_dir, f"task{task_id}")


def web_task_file(*, parameters: dict) -> str:
    """WebVoyager's tasks without the excluded sites."""
    return os.path.join(eval_root(parameters=parameters, env="web"), "WebVoyager_data.cusi.jsonl")


def web_source_task_file(*, parameters: dict) -> str:
    """The WebVoyager submodule's own task file."""
    return os.path.join(parameters["project_root"], "WebVoyager", "data", "WebVoyager_data.jsonl")


def web_native_root(*, parameters: dict) -> str:
    """Where WebVoyager's own run.py writes its runs (--output_dir)."""
    return os.path.join(_storage(parameters), "eval", "web_native")


def web_native_run(*, parameters: dict, run: str) -> str:
    return os.path.join(web_native_root(parameters=parameters), run)


def web_parity_task_file(*, parameters: dict, job: str) -> str:
    """The task subset a Web parity job draws."""
    return os.path.join(_storage(parameters), "eval", f"web_parity_{job}_tasks.jsonl")


def android_parity_run(*, parameters: dict, run: str) -> str:
    return os.path.join(_storage(parameters), "eval", "android_parity", run)


def info_docs_dir(*, parameters: dict, game: str) -> str:
    return os.path.join(_storage(parameters), "eval", "info_docs", game)


def parametric_doc_file(*, parameters: dict, game: str, model_name: str) -> str:
    """A model's cached parametric-knowledge document for a game."""
    return os.path.join(info_docs_dir(parameters=parameters, game=game), f"parametric_{model_name.replace('/', '_')}.json")


def eval_subsets_dir(*, parameters: dict, name: str) -> str:
    return os.path.join(_storage(parameters), "eval", f"{name}_subsets")


# --------------------------------------------------------------------------- practice

class PracticePaths:
    """The practice pipeline's directory for one env, model and task source ("zeroshot" or "curiosity")."""

    def __init__(self, *, parameters: dict, env_name: str, model_name: str, source: str = "zeroshot") -> None:
        self.root = practice_root(parameters=parameters, env=env_name, model_name=model_name, source=source)
        os.makedirs(self.root, exist_ok=True)

    @property
    def proposals(self) -> str:
        return os.path.join(self.root, "proposals.jsonl")

    @property
    def attempts_dir(self) -> str:
        return os.path.join(self.root, "attempts")

    @property
    def guidance(self) -> str:
        return os.path.join(self.root, "guidance.json")

    @property
    def practice_dir(self) -> str:
        return os.path.join(self.root, "practice")

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.root, "dataset")


def practice_root(*, parameters: dict, env: str, model_name: str, source: str = "zeroshot") -> str:
    root = os.path.join(_storage(parameters), "practice", env, _model_short(model_name))
    return os.path.join(root, "curiosity") if source == "curiosity" else root


# --------------------------------------------------------------------------- explore

def explore_root(*, parameters: dict, env: str) -> str:
    return os.path.join(_storage(parameters), "explore", env)


def explore_run(*, parameters: dict, env: str, run: str) -> str:
    return os.path.join(explore_root(parameters=parameters, env=env), run)


def explore_runs(*, parameters: dict, env: str, run_names: str) -> list:
    """Comma-separated run names -> their run directories."""
    return [explore_run(parameters=parameters, env=env, run=r.strip()) for r in run_names.split(",") if r.strip()]


def explore_replay(*, run_dir: str) -> str:
    return os.path.join(run_dir, "replay")


def runs_label(*, run_names: str) -> str:
    """Default output name for a model trained on several runs, e.g. "a+b"."""
    return run_names.replace(",", "+").replace("/", "_")


def world_model_dir(*, parameters: dict, env: str, name: str) -> str:
    return os.path.join(explore_root(parameters=parameters, env=env), "world_model", name)


def decoder_dir(*, parameters: dict, env: str, name: str) -> str:
    return os.path.join(explore_root(parameters=parameters, env=env), "decoder", name)


def embedder_dir(*, parameters: dict, env: str, embedder: str, name: str) -> str:
    return os.path.join(explore_root(parameters=parameters, env=env), "embedder", f"{embedder}_{name}")


def elements_file(*, parameters: dict, env: str, run_names: str) -> str:
    return os.path.join(explore_root(parameters=parameters, env=env), f"elements_{run_names.replace(',', '+')}.json")


# --------------------------------------------------------------------------- search

def search_root(*, parameters: dict, env: str) -> str:
    return os.path.join(_storage(parameters), "search", env)


def search_run(*, parameters: dict, env: str, run: str) -> str:
    return os.path.join(search_root(parameters=parameters, env=env), run)


# --------------------------------------------------------------------------- Bash access

def _main(argv: list) -> None:
    """Print ``<function>(--arg value ...)``, loading parameters if the function takes them."""
    if not argv or argv[0].startswith("_") or not callable(globals().get(argv[0])):
        names = sorted(n for n, f in globals().items() if callable(f) and not n.startswith("_")
                       and f.__module__ == __name__ and not isinstance(f, type))
        raise SystemExit(f"usage: python -m cusi.utils.paths <function> --arg value ...; functions: {names}")
    fn, kwargs = globals()[argv[0]], {}
    rest = argv[1:]
    if len(rest) % 2:
        raise SystemExit("arguments come in --name value pairs")
    for key, value in zip(rest[::2], rest[1::2]):
        kwargs[key.lstrip("-")] = value
    if "parameters" in fn.__code__.co_varnames[:fn.__code__.co_kwonlyargcount]:
        kwargs["parameters"] = load_parameters()
    print(fn(**kwargs))


if __name__ == "__main__":
    _main(sys.argv[1:])
