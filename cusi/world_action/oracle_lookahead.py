"""
The oracle lookahead: the candidate frames are the real screens that follow each button, taken from the emulator
itself with GameBoyWorlds' peek_high_level_action (save the PyBoy state, run the button through the controller as a
step does, read the screen, restore the emulator). The upper bound of "a VLM picking among outcomes": a learned world
model can at best match it. A diagnostic model: it needs the env (attach_env), which the executor passes it; no other
model may.

The peek runs with the emulator's listeners and video off and puts back its step counter, so it costs no benchmark
steps and changes no tracked state. PyBoy's save / load state round trip is exact, so the candidate for the action
then taken should match the real next screen (predicted_pixel_error 0): a non-zero value is a bug or non-determinism
and is worth a look.
"""
from typing import Optional
import numpy as np
from cusi.world_action.lookahead import EXACT_ARE, LookaheadModel


class OracleLookaheadModel(LookaheadModel):
    name = "oracle_lookahead"
    candidates_are = EXACT_ARE

    def __init__(self, *, parameters: dict) -> None:
        super().__init__(directory=None, meta={"env": "gameboy", "game": None, "game_series": None,
                                                "oracle": True}, parameters=parameters)
        self._env = None

    @classmethod
    def load(cls, *, directory: Optional[str], device: str, parameters: dict) -> "OracleLookaheadModel":
        return cls(parameters=parameters)

    def attach_env(self, env) -> None:
        self._env = env
        self.meta["game"] = getattr(env, "game", None)
        from cusi.world_action.base import game_series
        self.meta["game_series"] = game_series(game=self.meta["game"])

    def candidate_frames(self, *, obs: dict, info: dict) -> tuple:
        if self._env is None:
            raise RuntimeError("oracle_lookahead: attach_env was not called")
        from gameboy_worlds.interface.action import LowLevelAction, LowLevelActions
        by_name = {LowLevelAction.get_action_name(a): a for a in LowLevelActions}
        frames = []
        for name in self.candidates:
            frame = self._env._env.peek_high_level_action(LowLevelAction, low_level_action=by_name[name])
            if frame is None:
                raise RuntimeError(f"oracle_lookahead: the peek of {name} returned no frame")
            frames.append(np.asarray(frame, dtype=np.uint8).copy())
        return frames, {}
