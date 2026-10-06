"""
GameBoyWorlds as a TextActionEnv: one game at a fixed savestate, deterministic on reset.
"""
from enum import Enum
from typing import Any, Optional
import numpy as np
from gameboy_worlds import get_environment, get_test_environment
from gameboy_worlds.interface.action import LowLevelAction, LowLevelActions
from gameboy_worlds.utils import get_benchmark_tasks
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info, log_error
from cusi.envs.base import TextActionEnv, scene_hash

# GameBoyWorlds' OCR gives text regions as images (info["text_regions"]), not text.
# TODO: add an OCR text channel once the GameBoy OCR approach is decided.
TEXT_KEYS = ()

MSG_PARSE_FAILED = "Could not parse an action from the output. Reply with one of the allowed actions."
MSG_UNAVAILABLE = "That action is not available right now. No action was performed."


def render_action_strings(*, action_strings: Any) -> str:
    """get_action_strings() (a dict, or a bare string for some controllers) as one text block."""
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


def action_display_name(*, action_class: type, kwargs: dict) -> str:
    """The action's name as GameBoyRL's reports show it."""
    try:
        return action_class.get_action_name(**kwargs)
    except Exception:
        return action_class.__name__


BUTTONS = tuple(LowLevelAction.get_action_name(a) for a in LowLevelActions)


class GameBoyPlayEnv(TextActionEnv):
    """info also carries what GameBoyRL's executor records per step."""

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
        wait_ticks: Optional[int] = None,
        benchmark_row: Optional[dict] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """mode="test" needs benchmark_row (use from_benchmark); evaluation should pass wait_ticks=20."""
        self._parameters = load_parameters(parameters)
        emulator_kwargs = {"headless": True, "save_video": False, "session_name": session_name}
        if max_steps is not None:
            emulator_kwargs["max_steps"] = max_steps
        if wait_ticks is not None:
            emulator_kwargs["wait_ticks"] = wait_ticks
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
        self.env_description = f"the GameBoy game {game}"
        self.init_state = init_state
        self.scene_id = scene_hash(parts=("gameboy", game, init_state, mode,
                                          benchmark_row["task"] if benchmark_row is not None else None))
        # Already reset by GameBoyWorlds; resetting again would inflate info["core"]["steps"].
        raw_obs, raw_info = self._env.get_observation(), self._env.get_info()
        frame = self._frame(raw_obs=raw_obs)
        super().__init__(mode=mode, frame_shape=frame.shape, text_keys=TEXT_KEYS, parameters=self._parameters)
        self._last = (frame, raw_info)
        self._steps = 0
        self._fresh = True       # untouched since the last reset
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
        wait_ticks: Optional[int] = None,
        session_name: Optional[str] = None,
        parameters: dict[str, Any] = None,
    ) -> "GameBoyPlayEnv":
        """A benchmark task by row index or task text; free_play gives its start state without the goal."""
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
        common = dict(controller_variant=controller_variant, max_steps=max_steps, wait_ticks=wait_ticks,
                      session_name=session_name, parameters=parameters)
        if mode == "test":
            return cls(game=game, mode="test", benchmark_row=row, **common)
        return cls(game=game, init_state=row["init_state"], mode=mode, **common)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _frame(*, raw_obs: np.ndarray) -> np.ndarray:
        frame = np.asarray(raw_obs, dtype=np.uint8)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        return frame

    def action_strings(self, *, return_all: bool = False) -> dict:
        """The controller's {HighLevelAction class: description}, unrendered."""
        return self._env.get_action_strings(return_all=return_all)

    def sample_action(self) -> str:
        return str(BUTTONS[int(self.np_random.integers(len(BUTTONS)))])

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
        # Skip resetting an untouched emulator, so info["core"]["steps"] matches GameBoyRL's single reset.
        if self._fresh:
            raw_obs, raw_info = self._env.get_observation(), self._env.get_info()
        else:
            raw_obs, raw_info = self._env.reset()
        self._fresh = True
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
        self._fresh = False
        core_frame_before = self._env.get_info()["core"]["current_frame"]
        raw_obs, _reward, terminated, truncated, raw_info = self._env.step_high_level_action(action_class, **kwargs)
        frame = self._frame(raw_obs=raw_obs)
        self._last = (frame, raw_info)
        unavailable = bool(raw_info.get("invalid_action"))
        core = raw_info.get("core", {})
        if "previous_action_details" in core:
            action_success = core["previous_action_details"][3]
        else:
            action_success = -1
        if self.mode == "test":
            reward = 1.0 if terminated else 0.0   # benchmark success = the test tracker terminated
        else:
            reward, terminated = 0.0, False
        info = self.make_info(valid=not unavailable, error=MSG_UNAVAILABLE if unavailable else None,
                              parsed_action=parsed, step=self._steps, scene_id=self.scene_id,
                              action_name=action_display_name(action_class=action_class, kwargs=kwargs),
                              frame_changed=bool(core.get("frame_changed", True)),
                              action_success=action_success,
                              low_level=issubclass(action_class, LowLevelAction),
                              # GameBoyRL's EnvironmentStepRecord fields.
                              action_class=action_class, action_kwargs=dict(kwargs),
                              transition_states=(core["previous_action_details"][2]
                                                 if "previous_action_details" in core else []),
                              core_frame_before=core_frame_before, core_frame_after=core.get("current_frame"),
                              **self._extra(raw_info=raw_info))
        return self._obs(frame=frame), reward, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------ saved states
    # load_custom_state overwrites the emulator's init_state; it is restored so reset() returns to the scene start.

    def _save_state_impl(self, *, state_id: str) -> None:
        self._env.save_custom_state(state_id)

    def _load_state_impl(self, *, state_id: str) -> dict:
        self._fresh = False
        scene_init_state = self._env._emulator.init_state
        try:
            self._env.load_custom_state(state_id)
        finally:
            self._env._emulator.init_state = scene_init_state
        raw_obs, raw_info = self._env.get_observation(), self._env.get_info()
        frame = self._frame(raw_obs=raw_obs)
        self._last = (frame, raw_info)
        self._steps = 0
        return self._obs(frame=frame)

    def _delete_state_impl(self, *, state_id: str) -> None:
        self._env.delete_custom_state(state_id)

    def close(self) -> None:
        self.delete_all_states()
        self._env.close()
