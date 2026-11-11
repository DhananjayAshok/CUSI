"""
Optional server-side record of every session (off unless the server gets --record_dir): what the client was sent and
what it was not (true_state, the held-out checks), whatever the session asked for.

    <record_dir>/<session id>/session.json   how it was opened (task or scene, policy, state requested)
    <record_dir>/<session id>/steps.jsonl    one line per reset/step: action, the full info (true state included),
                                             the env's full_state(), reward, done, success; frames by file name
    <record_dir>/<session id>/frames/<n>.png the obs frame after each reset/step
"""
import json
import os
from PIL import Image
from cusi.env_server.codec import encode


class Recorder:
    def __init__(self, *, directory: str, session_id: str, meta: dict) -> None:
        self.directory = os.path.join(directory, session_id)
        os.makedirs(os.path.join(self.directory, "frames"), exist_ok=True)
        with open(os.path.join(self.directory, "session.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        self._n = 0

    def write(self, *, kind: str, action, obs: dict, info: dict, reward, done: bool, success, state) -> None:
        name = f"{self._n:05d}.png"
        frame = obs.get("frame")
        if frame is not None:
            img = frame[:, :, 0] if frame.ndim == 3 and frame.shape[2] == 1 else frame
            Image.fromarray(img).save(os.path.join(self.directory, "frames", name))
        # images and GameBoyRL's per-transition records (every intermediate frame) are left out
        heavy = {k for k, v in info.items() if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2} | {"transition_states"}
        row = {"n": self._n, "kind": kind, "action": action, "frame": name if frame is not None else None,
               "texts": obs.get("texts"), "reward": reward, "done": done, "success": success,
               "info": encode({k: v for k, v in info.items() if k not in heavy}), "state": encode(state)}
        with open(os.path.join(self.directory, "steps.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self._n += 1
