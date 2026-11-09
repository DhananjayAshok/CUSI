"""
The oracle lookahead: the candidate frames are the real screens that follow each button, taken from the emulator
itself (save the PyBoy state, press each button as GameBoyWorlds' emulator does, read the screen, restore). The upper
bound of "a VLM picking among outcomes": a learned world model can at best match it. A diagnostic model: it needs the
env (attach_env), which the executor passes it; no other model may.

The lookahead runs on PyBoy directly, below GameBoyWorlds' step counters, trackers and OCR, so it costs no benchmark
steps and changes no tracked state. PyBoy's save / load state round trip is exact, so the candidate for the action
then taken should match the real next screen (predicted_pixel_error 0): a non-zero value is a bug or non-determinism
and is worth a look.
"""
import io
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
        from gameboy_worlds.emulation.emulator import ReleaseActions
        from gameboy_worlds.interface.action import LowLevelAction, LowLevelActions
        emu = self._env._env._emulator
        pyboy = emu._pyboy
        by_name = {LowLevelAction.get_action_name(a): a for a in LowLevelActions}
        saved = io.BytesIO()
        pyboy.save_state(saved)
        frames = []
        for name in self.candidates:
            action = by_name[name]
            pyboy.send_input(action.value)
            pyboy.tick(emu.press_step, True)
            pyboy.send_input(ReleaseActions.release_actions.value[action])
            pyboy.tick(emu.wait_ticks + emu.press_step + 1, True)
            frames.append(np.asarray(emu.get_current_frame(), dtype=np.uint8).copy())
            saved.seek(0)
            pyboy.load_state(saved)
        saved.seek(0)
        pyboy.load_state(saved)
        return frames, {}
