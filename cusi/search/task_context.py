"""
The train-task context of tree annotation (cusi.search.annotate): a fixed, seeded sample of the benchmark's
train tasks, grouped by flavour, shown in every call so that proposed tasks take the train tasks' kind and form.

Only split == "train" tasks of configs/splits/benchmark_splits.json are ever read (checked again when the prompt
block is built). Flavours, per env:
- gameboy: the benchmark CSV's task_category, per game (a tree's game: its config's "game", else Pokemon Red);
- web: the site (WebVoyager web_name);
- android: the app (the task class's app prefix).

configs/annotation/<env>_train_flavours.json holds every flavour's train task ids and texts (written by `build`,
read by everything else), so every tree is annotated against the same lists:

    python -m cusi.search.task_context build
"""
import hashlib
import json
import os
import random
import re
import sys
from cusi.eval.splits import load_splits
from cusi.utils.paths import benchmark_splits_file

FLAVOUR_DIR = ("configs", "annotation")
ENVS = ("gameboy", "web", "android")
#: Android task-class prefixes that are two words (SimpleCalendar...) or name a combined system task.
_TWO_WORD = ("Simple", "Sports", "Audio")


def flavours_file(*, env: str, parameters: dict) -> str:
    return os.path.join(parameters["project_root"], *FLAVOUR_DIR, f"{env}_train_flavours.json")


def android_app(*, task_class: str) -> str:
    """The app an AndroidWorld train task class belongs to, from its CamelCase prefix."""
    if task_class.startswith("Turn"):
        return "System"
    words = re.findall(r"[A-Z][a-z]*", task_class)
    return "".join(words[:2]) if words[0] in _TWO_WORD else words[0]


def build_flavours(*, env: str, parameters: dict) -> dict:
    """{group: {flavour: [{"id", "task"}]}} from the split's train tasks; group is the game (GameBoy) or "all"."""
    splits = load_splits(parameters=parameters)
    train = {t: v["text"] for t, v in splits.items() if v["split"] == "train"}
    out = {}
    if env == "gameboy":
        from gameboy_worlds.utils import get_benchmark_tasks
        for tid, text in train.items():
            if not tid.startswith("gb."):
                continue
            game, row = tid[3:].rsplit(".", 1)
            category = str(get_benchmark_tasks(game=game).reset_index(drop=True).iloc[int(row)]["task_category"])
            out.setdefault(game, {}).setdefault(category.strip(), []).append({"id": tid, "task": text})
    elif env == "web":
        for tid, text in train.items():
            if tid.startswith("web."):
                site = tid[4:].split("--")[0]
                out.setdefault("all", {}).setdefault(site, []).append({"id": tid, "task": text})
    else:
        for tid, text in train.items():
            if tid.startswith("aw."):
                out.setdefault("all", {}).setdefault(android_app(task_class=tid[3:]), []).append(
                    {"id": tid, "task": text})
    return {g: {f: sorted(v, key=lambda x: x["id"]) for f, v in sorted(fl.items())} for g, fl in sorted(out.items())}


def split_hash(*, parameters: dict) -> str:
    with open(benchmark_splits_file(parameters=parameters), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


class TrainContext:
    def __init__(self, *, env: str, group: str, n_per_flavour: int, seed: int, parameters: dict) -> None:
        """group: the game (GameBoy) or "all"; n_per_flavour train tasks are drawn per flavour (all if fewer), once,
        by seed, so every call and every tree with these settings sees the same examples."""
        path = flavours_file(env=env, parameters=parameters)
        with open(path) as f:
            all_flavours = json.load(f)
        if group not in all_flavours:
            raise ValueError(f"{path} has no train tasks for {group!r} (only {sorted(all_flavours)}): its trees "
                             f"cannot be annotated")
        flavours = all_flavours[group]
        splits = load_splits(parameters=parameters)
        rng = random.Random(f"{seed}:{env}:{group}")
        self.sample = {}
        for flavour, tasks in flavours.items():
            for t in tasks:
                if splits[t["id"]]["split"] != "train":
                    raise ValueError(f"{path}: {t['id']} is {splits[t['id']]['split']}, not train")
            picked = tasks if len(tasks) <= n_per_flavour else rng.sample(tasks, n_per_flavour)
            self.sample[flavour] = sorted(picked, key=lambda x: x["id"])
        self.meta = {"env": env, "group": group, "n_per_flavour": n_per_flavour, "seed": seed,
                     "split_hash": split_hash(parameters=parameters),
                     "sample": {f: [t["id"] for t in v] for f, v in self.sample.items()}}

    def block(self) -> str:
        """The prompt block: the sampled train tasks under their flavour."""
        lines = []
        for flavour, tasks in self.sample.items():
            lines.append(f"[{flavour}]")
            lines += [f"- {' '.join(t['task'].split())}" for t in tasks]
        return "\n".join(lines)


def tree_group(*, env: str, config: dict) -> str:
    """A tree's flavour group: its game on GameBoy, else "all"."""
    if env != "gameboy":
        return "all"
    from cusi.agents.specs import GAMEBOY_GAME
    return config.get("game") or GAMEBOY_GAME


if __name__ == "__main__":
    from cusi.utils.parameter_handling import load_parameters
    if sys.argv[1:] != ["build"]:
        sys.exit("usage: python -m cusi.search.task_context build")
    params = load_parameters()
    os.makedirs(os.path.join(params["project_root"], *FLAVOUR_DIR), exist_ok=True)
    for e in ENVS:
        fl = build_flavours(env=e, parameters=params)
        with open(flavours_file(env=e, parameters=params), "w") as f:
            json.dump(fl, f, indent=1)
        print(e, {g: {k: len(v) for k, v in x.items()} for g, x in fl.items()})
