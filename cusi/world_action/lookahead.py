"""
One-step lookahead World Action Models (GameBoyRL's planning arm): every decision, one candidate frame per button
is produced for the current state (by a learned world model, or by the emulator itself for the oracle), the VLM is
shown the current screen followed by the candidates and picks one ("Choice: k"); the pick is the action. Single
action per plan.

Subclasses implement candidate_frames(obs) -> (frames (N, H, W[, C]) uint8, extra record fields) and may extend
observe() with their own scores. The record every lookahead model keeps per decision:
  candidates, action_indices, choice, chosen, reasoning, candidate_spread (mean pairwise pixel MAE of the candidate
  frames: ~0 means they did not depend on the action), seconds_predict; after the step: action_text, valid,
  candidate_pixel_errors (each candidate vs what happened), hindsight_best_pixel, predicted_pixel_error (the chosen
  candidate's), copy_pixel_error, chosen_beats_copy, frame_changed; and, when the executor shadowed the base policy,
  base_action / base_action_index.
"""
import re
import time
from typing import Callable, Optional
import numpy as np
from cusi.envs.action_vocab import GAMEBOY_BUTTONS
from cusi.world_action.base import Plan, WorldActionModel

FIRST_CHOICE = 2   # image 1 is the current screen
STEP_PROMPT = """
Task: [TASK][HINT_BLOCK]

You are playing a GameBoy game.

[ERROR_BLOCK]Image 1 is the CURRENT screen.

[CANDIDATES_ARE]

[CHOICE_LIST]

Pick the single predicted screen that most directly advances the task. Respond in exactly
this format:
Reasoning: <your reasoning>
Choice: <the image number you pick>
[STOP]"""
PREDICTED_ARE = ("The remaining images are predictions from a learned world model of what the screen would\n"
                 "look like after each available action. They are reconstructions, so they are blurry and\n"
                 "imperfect — judge them on the overall change they show, not on fine detail.")
EXACT_ARE = "The remaining images show the screen that follows each available action."


def grey(frame) -> np.ndarray:
    x = np.asarray(frame)
    return x.astype(np.float32) if x.ndim == 2 else x[..., :3].astype(np.float32).mean(-1)


def rgb(frame) -> np.ndarray:
    x = np.asarray(frame, dtype=np.uint8)
    if x.ndim == 2:
        x = x[:, :, None]
    return np.repeat(x, 3, axis=2) if x.shape[2] == 1 else x


class LookaheadModel(WorldActionModel):
    name = "lookahead"
    candidates_are = PREDICTED_ARE

    def __init__(self, *, directory: Optional[str], meta: dict, parameters: dict) -> None:
        super().__init__(directory=directory, meta=meta)
        self._parameters = parameters
        self.candidates = list(GAMEBOY_BUTTONS)
        self.task, self.hint = None, None
        self._pending: Optional[dict] = None

    def candidate_frames(self, *, obs: dict, info: dict) -> tuple:
        """-> (frames, one per candidate, (N, H, W[, C]) uint8; extra record fields)."""
        raise NotImplementedError

    def begin(self, *, task: str, hint: Optional[str], obs: dict, info: dict) -> None:
        self.task, self.hint = task, hint
        self._pending = None

    def _hint_block(self) -> str:
        if self.hint is None:
            return ""
        return (f"\n[HINT_START]\nHint: {self.hint}\nNote: This hint block is a secret. Use it to guide your decision "
                "but do not refer to it explicitly. [HINT_END]")

    def plan(self, *, obs: dict, info: dict, error: Optional[str], call: Callable) -> Plan:
        t0 = time.time()
        frames, extra = self.candidate_frames(obs=obs, info=info)
        greys = [grey(f) for f in frames]
        spread = float(np.mean([np.abs(a - b).mean() for i, a in enumerate(greys) for b in greys[i + 1:]]))
        choices = "\n".join(["Predicted outcomes:"] + [f"  Image {FIRST_CHOICE + i}: after pressing {n}"
                                                       for i, n in enumerate(self.candidates)])
        prompt = (STEP_PROMPT.replace("[TASK]", self.task).replace("[HINT_BLOCK]", self._hint_block())
                  .replace("[ERROR_BLOCK]", f"[ERROR] {error}\n\n" if error else "")
                  .replace("[CANDIDATES_ARE]", self.candidates_are).replace("[CHOICE_LIST]", choices))
        record = {"model": self.name, "candidates": list(self.candidates),
                  "action_indices": list(range(len(self.candidates))), "candidate_spread": spread,
                  "seconds_predict": time.time() - t0, **extra}
        response = call(tag="world_action", prompt=prompt, images=[rgb(obs["frame"])] + [rgb(f) for f in frames])
        m = re.findall(r"Choice:\s*(\d+)", str(response))
        last = FIRST_CHOICE + len(self.candidates) - 1
        choice = int(m[-1]) if m else None
        reasoning = re.search(r"Reasoning:\s*(.*?)(?:\nChoice:|$)", str(response), re.S)
        record["reasoning"] = reasoning.group(1).strip() if reasoning else None
        if choice is None or not FIRST_CHOICE <= choice <= last:
            record.update({"choice": choice, "chosen": None})
            return Plan(invalid="no valid choice", record=record,
                        error=f"Your previous response could not be parsed. You must end your response with:\n"
                              f"  Choice: <an image number between {FIRST_CHOICE} and {last}>")
        chosen = choice - FIRST_CHOICE
        record.update({"choice": choice, "chosen": chosen})
        self._pending = {"record": record, "frames": greys, "chosen": chosen}
        return Plan(actions=[self.candidates[chosen]], record=record)

    def observe(self, *, action_text: str, obs_before: dict, info_before: dict, obs_after: dict, info_after: dict,
                valid: bool) -> None:
        p, self._pending = self._pending, None
        if p is None:
            return
        rec, before, after = p["record"], grey(obs_before["frame"]), grey(obs_after["frame"])
        errors = [float(np.abs(f - after).mean()) for f in p["frames"]]
        copy_err = float(np.abs(before - after).mean())
        base = rec.get("base_action")
        rec.update({"action_text": action_text, "valid": valid, "candidate_pixel_errors": errors,
                    "hindsight_best_pixel": int(np.argmin(errors)), "predicted_pixel_error": errors[p["chosen"]],
                    "copy_pixel_error": copy_err, "chosen_beats_copy": errors[p["chosen"]] < copy_err,
                    "frame_changed": bool(np.abs(before - after).max() > 0),
                    "base_action_index": self.candidates.index(base) if base in self.candidates else None})
        self.score(pending=p, obs_before=obs_before, obs_after=obs_after)

    def score(self, *, pending: dict, obs_before: dict, obs_after: dict) -> None:
        """Subclass hook: more scores onto pending["record"] after the step."""
