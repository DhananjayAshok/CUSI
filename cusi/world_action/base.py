"""
The World Action Model (WAM) contract: the env's state in, the actions to execute out. Everything about how a model
arrives at them (embeddings, predicted frames, a VLM picking among them, rollouts) stays inside the model; the
executor (cusi/agents/executors/world_action.py) only executes what plan() returns, one env step per action, and
reports each step back through observe().

A model directory holds the model and its metadata; the metadata names the env, the game the model was trained on
and that game's series, so an eval on another game of the series (pokemon_red -> pokemon_crystal) is recorded as a
transfer. load() refuses a model for another env.
"""
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: GameBoyWorlds game -> its series; a game not listed is its own series.
GAME_SERIES = {"pokemon_red": "pokemon", "pokemon_crystal": "pokemon",
               "legend_of_zelda_links_awakening": "zelda", "legend_of_zelda_the_oracle_of_seasons": "zelda",
               "deja_vu_1": "deja_vu", "deja_vu_2": "deja_vu", "sword_of_hope_1": "sword_of_hope",
               "sword_of_hope_2": "sword_of_hope", "bomberman_quest": "bomberman", "bomberman_max": "bomberman"}


def game_series(*, game: Optional[str]) -> Optional[str]:
    return None if game is None else GAME_SERIES.get(game, game)


@dataclass
class Plan:
    """What the model wants executed next. Exactly one of: actions (executed in order, one env step each), finish,
    or invalid (the model could not decide; `error` is shown to it next time, as an executor's parse failure)."""
    actions: list = field(default_factory=list)      # action strings in the env's own format
    finish: bool = False
    answer: Optional[str] = None
    invalid: Optional[str] = None
    error: Optional[str] = None
    #: Whatever the model wants kept with the deciding call (CallRecord.extra["world_action"]); the model may keep
    #: filling it in from observe(), so post-run diagnostics see the prediction and what actually happened together.
    record: dict = field(default_factory=dict)


class WorldActionModel:
    """Subclasses implement load, begin, plan and observe; history / restore_history if they keep state across legs.

    `call(tag=, prompt=, images=) -> response` is the executor's recorded VLM call: a model that asks a VLM uses it,
    so its calls are CallRecords like any executor's."""

    name = "base"

    def __init__(self, *, directory: str, meta: dict) -> None:
        self.directory = directory
        self.meta = meta

    # ------------------------------------------------------------ identity

    @property
    def env_name(self) -> str:
        return self.meta["env"]

    @property
    def game(self) -> Optional[str]:
        return self.meta.get("game")

    @property
    def series(self) -> Optional[str]:
        return self.meta.get("game_series") or game_series(game=self.game)

    def transfer_kind(self, *, game: Optional[str]) -> str:
        """How the eval's game relates to the training game: same_game, same_series or other_series."""
        if game is None or self.game is None:
            return "unknown"
        if game == self.game:
            return "same_game"
        return "same_series" if game_series(game=game) == self.series else "other_series"

    def summary(self) -> dict:
        """Recorded in run configs and episode metadata."""
        return {"model": self.name, "directory": self.directory, "env": self.env_name, "game": self.game,
                "game_series": self.series}

    # ------------------------------------------------------------ the contract

    @classmethod
    def load(cls, *, directory: str, device: str, parameters: dict) -> "WorldActionModel":
        raise NotImplementedError

    def begin(self, *, task: str, hint: Optional[str], obs: dict, info: dict) -> None:
        """Start of a leg from the current state. State restored by restore_history is kept."""

    def plan(self, *, obs: dict, info: dict, error: Optional[str], call: Callable) -> Plan:
        raise NotImplementedError

    def observe(self, *, action_text: str, obs_before: dict, info_before: dict, obs_after: dict,
                info_after: dict, valid: bool) -> None:
        """What one executed action did; called once per env step, after it."""

    def attach_env(self, env) -> None:
        """Diagnostic models only (the oracle lookahead reads the emulator): the executor passes its env at begin().
        A real world action model never needs this."""

    def history(self) -> Any:
        """State that must outlive a leg (the executor is rebuilt per leg)."""
        return None

    def restore_history(self, history: Any) -> None:
        pass


def read_meta(*, directory: str) -> dict:
    import json
    path = os.path.join(directory, "world_model_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no world_model_meta.json in {directory}")
    with open(path) as f:
        return json.load(f)
