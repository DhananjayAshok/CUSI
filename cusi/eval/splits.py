"""
The benchmark split: configs/splits/benchmark_splits.json, {task id: {"split", "text"}} for every task of the three
benchmarks. Splits: train, val, test (held-out tasks of the train envs), test_env (every task of the held-out envs).

Task ids: gb.<game>.<row:03d> (GameBoyWorlds benchmark row), web.<WebVoyager id>, aw.<AndroidWorld task class> (the
text is its seed-0 instance). resolve() turns an id into what that benchmark's env needs; nothing else in the split
knows how a benchmark is wired.

    python -m cusi.eval.splits --env gameboy --splits train,val
"""
import json
import sys
from functools import lru_cache
from cusi.utils.log_handling import log_error
from cusi.utils.paths import benchmark_splits_file, web_source_task_file

SPLITS = ("train", "val", "test", "test_env")
PREFIX = {"gb": "gameboy", "web": "web", "aw": "android"}


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_splits(*, parameters: dict) -> dict:
    """{task id: {"split", "text"}}."""
    return _load(benchmark_splits_file(parameters=parameters))


def env_of(*, task_id: str) -> str:
    """gameboy, web or android, from the id's prefix."""
    return PREFIX[task_id.split(".", 1)[0]]


def task_ids(*, splits: tuple, parameters: dict, env: str = None) -> list:
    """The ids in any of splits (optionally one env's), in file order."""
    bad = set(splits) - set(SPLITS)
    if bad:
        log_error(f"unknown splits {sorted(bad)}; splits are {SPLITS}", parameters=parameters)
    return [t for t, v in load_splits(parameters=parameters).items()
            if v["split"] in splits and (env is None or env_of(task_id=t) == env)]


def split_of(*, task_id: str, parameters: dict) -> str:
    entry = load_splits(parameters=parameters).get(task_id)
    if entry is None:
        log_error(f"{task_id!r} is not in {benchmark_splits_file(parameters=parameters)}", parameters=parameters)
    return entry["split"]


@lru_cache(maxsize=None)
def _web_rows(path: str) -> dict:
    with open(path) as f:
        return {d["id"]: d for d in map(json.loads, f)}


def resolve(*, task_id: str, parameters: dict) -> dict:
    """What the id's env is built from:
    gameboy: {"env", "game", "task_index"} (GameBoyPlayEnv.from_benchmark);
    web: {"env", "row"} (the WebVoyager task row);
    android: {"env", "task_class", "instance"} (cusi.eval.android runs instance 0, the text's instance)."""
    split_of(task_id=task_id, parameters=parameters)
    prefix, rest = task_id.split(".", 1)
    if prefix == "gb":
        game, row = rest.rsplit(".", 1)
        return {"env": "gameboy", "game": game, "task_index": int(row)}
    if prefix == "web":
        return {"env": "web", "row": _web_rows(web_source_task_file(parameters=parameters))[rest]}
    return {"env": "android", "task_class": rest, "instance": 0}


if __name__ == "__main__":
    from cusi.utils.parameter_handling import load_parameters
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    ids = task_ids(splits=tuple(args.get("--splits", ",".join(SPLITS)).split(",")), env=args.get("--env"),
                   parameters=load_parameters())
    print("\n".join(ids))
