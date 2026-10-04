"""GameBoyWorlds as a TextActionEnv (see cusi_envs/base.py).

    # A fixed start state, as GameBoyRL's curiosity runs use (scripts/core_rl/train.sh):
    env = GameBoyPlayEnv(game="pokemon_red", init_state="location_viridian_city_starting_charmander")
    # A benchmark task (GameBoyWorlds/benchmark/tests/<game>.csv):
    env = GameBoyPlayEnv.from_benchmark(game="pokemon_red", task_index=0, mode="test")

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step("A")

The scene is one GameBoyWorlds savestate (`init_state`), fixed at construction.
reset() reloads it: a complete, deterministic restore of the emulator.

Observation: frame = the Game Boy screen as HxWx3 uint8 (grayscale repeated to 3
channels); texts = {} (GameBoyWorlds' OCR trackers capture text *regions* as images,
not text; those are passed on as info["text_regions"]); actions = the controller's
get_action_strings(); goal = the benchmark task in "test" mode, "" otherwise.

Modes (base.py):
    test       built with GameBoyWorlds' get_test_environment(row): the task's test
               tracker decides `terminated` (= success, as the benchmark scores it) and
               `truncated`; reward 1.0 on success, else 0.
    free_play  the game's "default" environment at the same kind of fixed start state:
               no goal, reward 0, never terminated; truncated only at max_steps.

Text that does not parse to an action (controller.string_to_high_level_action), or
that the controller reports as unavailable in the current state, is a no-op with
info["valid"] = False; invalid text does not advance the emulator.
"""
from enum import Enum
from typing import Any, Optional
import numpy as np
from gameboy_worlds import get_environment, get_test_environment
from gameboy_worlds.utils import get_benchmark_tasks
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_info, log_error
from cusi_envs.base import TextActionEnv, scene_hash

# GameBoyWorlds exposes no text channel (see the module docstring).
TEXT_KEYS = ()

MSG_PARSE_FAILED = "Could not parse an action from the output. Reply with one of the allowed actions."
MSG_UNAVAILABLE = "That action is not available right now. No action was performed."


def render_action_strings(*, action_strings: Any) -> str:
    """get_action_strings() returns {ActionClass: description} (or a bare string for
    some controllers); render it as one text block."""
    if isinstance(action_strings, dict):
        parts = [str(v).strip() for v in action_strings.values()]
    else:
        parts = [str(action_strings).strip()]
    return "\n".join(" ".join(p.split()) for p in parts if p)


def canonical_action(*, action_class: type, kwargs: dict) -> dict:
    """{"action_type": <HighLevelAction class name>, **kwargs}, with enums as names."""
    action = {"action_type": action_class.__name__}
    for key, value in kwargs.items():
        action[key] = value.name if isinstance(value, Enum) else value
    return action


class GameBoyPlayEnv(TextActionEnv):
    """A GameBoyWorlds game at a fixed savestate as a text-action gym.Env."""

    def __init__(
        self,
        *,
        game: str,
        init_state: Optional[str] = None,
        mode: str = "free_play",
        controller_variant: str = "low_level",
        environment_variant: str = "default",
        max_steps: Optional[int] = None,
        session_name: Optional[str] = None,
        save_video: bool = False,
        benchmark_row: Optional[dict] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        :param game: GameBoyWorlds game, e.g. "pokemon_red".
        :param init_state: Savestate name (gameboy_worlds get_available_init_states).
            None: the game's default start state.
        :param mode: "test" needs a benchmark row (use from_benchmark); "free_play".
        :param environment_variant: free_play only; "default" as the curiosity runs use.
        :param max_steps: Emulator step limit (truncation). None: GameBoyWorlds' default.
        :param benchmark_row: Set by from_benchmark.
        """
        self._parameters = load_parameters(parameters)
        emulator_kwargs = {"headless": True, "save_video": save_video, "session_name": session_name}
        if max_steps is not None:
            emulator_kwargs["max_steps"] = max_steps
        if mode == "test":
            if benchmark_row is None:
                log_error("mode='test' needs a benchmark task; use GameBoyPlayEnv.from_benchmark(...)",
                          parameters=self._parameters)
            self._env = get_test_environment(benchmark_row, controller_variant=controller_variant,
                                             **emulator_kwargs)
            self._goal = str(benchmark_row["task"])
            init_state = benchmark_row["init_state"]
        else:
            self._env = get_environment(game=game, environment_variant=environment_variant,
                                        init_state=init_state, controller_variant=controller_variant,
                                        **emulator_kwargs)
            self._goal = ""
        self.game = game
        self.init_state = init_state
        self.scene_id = scene_hash(parts=("gameboy", game, init_state, mode,
                                          benchmark_row["task"] if benchmark_row is not None else None))
        raw_obs, raw_info = self._env.reset()
        frame = self._frame(raw_obs=raw_obs)
        super().__init__(mode=mode, frame_shape=frame.shape, text_keys=TEXT_KEYS, parameters=self._parameters)
        self._last = (frame, raw_info)
        self._steps = 0
        log_info(f"GameBoyPlayEnv ready: {game} @ {init_state} (mode {mode})", parameters=self._parameters)

    @classmethod
    def from_benchmark(
        cls,
        *,
        game: str,
        task_index: Optional[int] = None,
        task: Optional[str] = None,
        mode: str = "test",
        controller_variant: str = "low_level",
        max_steps: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> "GameBoyPlayEnv":
        """Build the env for a GameBoyWorlds benchmark task, chosen by row index or by its
        task text. In free_play mode: the task's start state with no goal."""
        parameters = load_parameters(parameters)
        tasks = get_benchmark_tasks(game).reset_index(drop=True)
        if task is not None:
            matches = tasks[tasks["task"] == task]
            if len(matches) != 1:
                log_error(f"Expected one benchmark task {task!r} for {game}, found {len(matches)}",
                          parameters=parameters)
            row = matches.iloc[0]
        elif task_index is not None:
            row = tasks.iloc[task_index]
        else:
            log_error("Pass task_index or task", parameters=parameters)
        row = dict(row)
        if mode == "test":
            return cls(game=game, mode="test", controller_variant=controller_variant, max_steps=max_steps,
                       benchmark_row=row, parameters=parameters)
        return cls(game=game, init_state=row["init_state"], mode=mode, controller_variant=controller_variant,
                   max_steps=max_steps, parameters=parameters)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _frame(*, raw_obs: np.ndarray) -> np.ndarray:
        frame = np.asarray(raw_obs, dtype=np.uint8)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        return frame

    def _obs(self, *, frame: np.ndarray) -> dict:
        actions = render_action_strings(action_strings=self._env.get_action_strings())
        return {"frame": frame, "texts": {}, "actions": actions, "goal": self._goal}

    @staticmethod
    def _extra(*, raw_info: dict) -> dict:
        extra = {}
        ocr = raw_info.get("ocr")
        if ocr:
            extra["text_regions"] = ocr
        if "subgoals" in raw_info:
            extra["subgoals"] = raw_info["subgoals"]
        return extra

    def _parse(self, *, text: str):
        """The whole output, else what follows the last 'Action:', else its last line."""
        candidates = [text.strip()]
        if "Action:" in text:
            candidates.append(text.rsplit("Action:", 1)[1].strip())
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if lines:
            candidates.append(lines[-1])
        for candidate in candidates:
            action_class, kwargs = self._env.string_to_high_level_action(candidate)
            if action_class is not None:
                return action_class, kwargs or {}
        return None, None

    # ---------------------------------------------------------------- gym API

    def _reset_impl(self):
        raw_obs, raw_info = self._env.reset()
        frame = self._frame(raw_obs=raw_obs)
        self._last = (frame, raw_info)
        self._steps = 0
        info = self.make_info(valid=True, error=None, parsed_action=None, step=0, scene_id=self.scene_id,
                              **self._extra(raw_info=raw_info))
        return self._obs(frame=frame), info

    def _step_impl(self, action: str):
        action_class, kwargs = self._parse(text=action)
        self._steps += 1
        if action_class is None:
            frame, raw_info = self._last
            info = self.make_info(valid=False, error=MSG_PARSE_FAILED, parsed_action=None, step=self._steps,
                                  scene_id=self.scene_id)
            return self._obs(frame=frame), 0.0, False, False, info
        parsed = canonical_action(action_class=action_class, kwargs=kwargs)
        raw_obs, _reward, terminated, truncated, raw_info = self._env.step_high_level_action(action_class, **kwargs)
        frame = self._frame(raw_obs=raw_obs)
        self._last = (frame, raw_info)
        unavailable = bool(raw_info.get("invalid_action"))
        if self.mode == "test":
            reward = 1.0 if terminated else 0.0   # benchmark success = the test tracker terminated
        else:
            reward, terminated = 0.0, False
        info = self.make_info(valid=not unavailable, error=MSG_UNAVAILABLE if unavailable else None,
                              parsed_action=parsed, step=self._steps, scene_id=self.scene_id,
                              **self._extra(raw_info=raw_info))
        return self._obs(frame=frame), reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._env.close()
