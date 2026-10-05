"""Chunked on-disk replay buffer of exploration steps (plan Part 2.5).

Each transition (one env.step) stores: the frame reached (PNG; Android frames are 7.7 MB raw),
the active image embedder's embedding (float16) and the embedder's name, the text channels, the
policy's generated text, the canonical parsed_action, the world-model action index, valid, the
rewards (extrinsic; the curiosity components frame / region (reward_text is the same value, the
pre-migration name) / invalid-action penalty; total), and the episode boundaries (episode id,
step in episode, done). Code that needs another embedder re-embeds from the frames. The
first observation of every episode is stored as a step with step=0 and no action, so each
episode is self-contained: frame[t-1] -> action[t] -> frame[t].

    writer = ReplayWriter(directory=..., chunk_size=500)
    writer.add(**fields); writer.close()
    for step in iter_replay(directory=...): ...
    episodes = load_episodes(directory=...)       # list of lists of step dicts
"""
import glob
import os
import pickle
from typing import Iterator, Optional
import numpy as np
from cusi_practice.records import EncodedImage

FIELDS = ("episode", "step", "frame", "embedding", "texts", "generated", "parsed_action", "action_index", "valid",
          "reward_ext", "reward_frame", "reward_text", "reward", "done", "embedder", "reward_region",
          "reward_penalty")


class ReplayWriter:
    def __init__(self, *, directory: str, chunk_size: int = 500, start_chunk: Optional[int] = None) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.chunk_size = chunk_size
        existing = sorted(glob.glob(os.path.join(directory, "chunk_*.pkl")))
        self._chunk = start_chunk if start_chunk is not None else len(existing)
        self._rows: list = []
        self.n_written = 0

    def add(self, *, episode: int, step: int, frame, embedding, texts: dict, generated: Optional[str],
            parsed_action: Optional[dict], action_index: Optional[int], valid: bool, reward_ext: float,
            reward_frame: float, reward_text: float, reward: float, done: bool, embedder: Optional[str] = None,
            reward_penalty: float = 0.0) -> None:
        self._rows.append({
            "episode": int(episode), "step": int(step), "frame": EncodedImage.of(frame),
            "embedding": np.asarray(embedding.detach().cpu() if hasattr(embedding, "detach") else embedding,
                                    dtype=np.float16).reshape(-1),
            "texts": dict(texts), "generated": generated, "parsed_action": parsed_action,
            "action_index": action_index, "valid": bool(valid), "reward_ext": float(reward_ext),
            "reward_frame": float(reward_frame), "reward_text": float(reward_text), "reward": float(reward),
            "done": bool(done), "embedder": embedder, "reward_region": float(reward_text),
            "reward_penalty": float(reward_penalty)})
        if len(self._rows) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        path = os.path.join(self.directory, f"chunk_{self._chunk:05d}.pkl")
        with open(path + ".tmp", "wb") as f:
            pickle.dump(self._rows, f)
        os.replace(path + ".tmp", path)
        self.n_written += len(self._rows)
        self._chunk += 1
        self._rows = []

    def close(self) -> None:
        self.flush()


def iter_replay(*, directory: str, load_frames: bool = True) -> Iterator[dict]:
    """Steps of every chunk under `directory` (recursively), in order."""
    paths = sorted(glob.glob(os.path.join(directory, "**", "chunk_*.pkl"), recursive=True))
    for path in paths:
        with open(path, "rb") as f:
            rows = pickle.load(f)
        for row in rows:
            row["source"] = os.path.dirname(path)
            if not load_frames:
                row["frame"] = None
            yield row


def load_episodes(*, directory: str, load_frames: bool = True) -> list:
    """Episodes as lists of step dicts (step 0 = the reset observation), grouped by (run dir, episode)."""
    episodes: dict = {}
    for row in iter_replay(directory=directory, load_frames=load_frames):
        episodes.setdefault((row["source"], row["episode"]), []).append(row)
    out = []
    for key in sorted(episodes):
        steps = sorted(episodes[key], key=lambda r: r["step"])
        out.append(steps)
    return out
