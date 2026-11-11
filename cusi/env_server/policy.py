"""
What a server instance shows, fixed at launch (the client cannot change it):

    dev   train + val tasks, free play in the train envs' scenes, state (true_state) when a session asks, save/load
    test  test + test_env tasks (ids come from our harness; nothing is listed), never state, no save/load

Both return success when a task episode ends. Held-out envs: the GameBoy games, websites and Android apps whose tasks
are all test_env in the split. On dev, a step that would land on a held-out website or in a held-out app is undone by
the env before it is observed and is an invalid action (the envs' blocked lists); free play is also kept on its site or
app by the envs' scopes.
"""
from dataclasses import dataclass
from cusi.eval.splits import load_splits, env_of
from cusi.agents.specs import ANDROID_PACKAGES, ENV_SPECS

KINDS = ("dev", "test")
#: The held-out Android apps (every task touching one is test_env), as ANDROID_PACKAGES names.
HELD_OUT_APPS = ("osmand", "chrome", "expense", "vlc", "camera", "contacts", "draw", "gallery", "clipper")
#: The held-out websites, as ENV_SPECS["web"] scene names; their scope says what counts as being on them.
HELD_OUT_SITES = ("wolframalpha", "google_flights", "google_maps")
#: GameBoy games with train tasks (free play starts from their train and val tasks' start states).
TRAIN_GAMES = ("pokemon_red", "legend_of_zelda_links_awakening")


@dataclass(frozen=True)
class Policy:
    kind: str
    splits: tuple
    free_play: bool
    state: bool
    save_load: bool

    @classmethod
    def of(cls, *, kind: str) -> "Policy":
        if kind == "dev":
            return cls(kind=kind, splits=("train", "val"), free_play=True, state=True, save_load=True)
        if kind == "test":
            return cls(kind=kind, splits=("test", "test_env"), free_play=False, state=False, save_load=False)
        raise ValueError(f"unknown instance kind {kind!r}; one of {KINDS}")

    def visible(self, *, task_id: str, parameters: dict) -> bool:
        entry = load_splits(parameters=parameters).get(task_id)
        return entry is not None and entry["split"] in self.splits

    def listed(self, *, env: str, parameters: dict) -> list:
        """The tasks a client may list: dev's train + val; nothing on test."""
        if self.kind != "dev":
            return []
        return [{"task_id": t, "split": v["split"], "text": v["text"]}
                for t, v in load_splits(parameters=parameters).items()
                if v["split"] in self.splits and env_of(task_id=t) == env]


def held_out_packages() -> frozenset:
    return frozenset(ANDROID_PACKAGES[a] for a in HELD_OUT_APPS)


def held_out_scopes() -> list:
    return [ENV_SPECS["web"].scenes[s].kwargs["scope"] for s in HELD_OUT_SITES]


def free_play_scenes(*, env: str, parameters: dict) -> dict:
    """{scene name: env kwargs} a dev free-play session may start from.
    gameboy: "<game>/<init_state>" for the start states of the train games' train and val tasks (test tasks' start
    states are never shared with them: the split keeps a start state in one split);
    web: ENV_SPECS' website scenes minus the held-out sites; android: its app scenes minus the held-out apps, each
    scope without held-out packages."""
    if env == "gameboy":
        from gameboy_worlds.utils import get_benchmark_tasks
        splits = load_splits(parameters=parameters)
        scenes = {}
        for game in TRAIN_GAMES:
            rows = get_benchmark_tasks(game=game).reset_index(drop=True)
            for i, row in rows.iterrows():
                if splits[f"gb.{game}.{i:03d}"]["split"] in ("train", "val"):
                    scenes[f"{game}/{row['init_state']}"] = {"game": game, "init_state": row["init_state"]}
        return scenes
    if env == "web":
        return {name: dict(s.kwargs) for name, s in ENV_SPECS["web"].scenes.items() if name not in HELD_OUT_SITES}
    if env == "android":
        out, banned = {}, held_out_packages()
        for name, s in ENV_SPECS["android"].scenes.items():
            if name in HELD_OUT_APPS:
                continue
            kwargs = dict(s.kwargs)
            kwargs["scope"] = {"packages": [p for p in kwargs["scope"]["packages"] if p not in banned]}
            out[name] = kwargs
        return out
    raise ValueError(f"unknown env {env!r}")
