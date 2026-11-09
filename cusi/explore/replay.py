"""
Reading the chunked on-disk replays of exploration runs (written by the removed PPO explorer; search runs are read
by cusi/search/replay.py); step 0 of each episode is the reset observation.
"""
import glob
import hashlib
import os
import pickle
from typing import Iterator, Optional

FIELDS = ("episode", "step", "frame", "embedding", "texts", "generated", "parsed_action", "action_index", "valid",
          "reward_ext", "reward_frame", "reward_text", "reward", "done", "embedder", "reward_region",
          "reward_penalty")


#: Share of episodes held out by is_validation_episode (embedder, world model and wm_eval alike).
VAL_FRAC = 0.1


def is_validation_episode(*, source: str, episode, seed: int, val_frac: float = VAL_FRAC) -> bool:
    """Whether an episode is held out. A hash of (run dir, episode, seed), so every model trained on a replay (the
    SigLIP fine-tune, the world model on its embeddings, wm_eval) holds out the same episodes whatever it loads."""
    key = f"{seed}:{os.path.normpath(source)}:{episode}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2 ** 64 < val_frac


def iter_replay(*, directory: str, load_frames: bool = True, scene: Optional[str] = None,
                embed: bool = True) -> Iterator[dict]:
    """Steps of every chunk under `directory` (recursively), in order; a search run directory is read as replay
    (cusi/search/replay.py, one episode per expansion, optionally of one scene). embed=False skips recomputing a
    search run's step embeddings (for callers that re-embed the frames themselves)."""
    from cusi.search.replay import is_search_run, iter_search_steps
    if is_search_run(directory=directory):
        yield from iter_search_steps(directory=directory, load_frames=load_frames, scene=scene, embed=embed)
        return
    paths = sorted(glob.glob(os.path.join(directory, "**", "chunk_*.pkl"), recursive=True))
    for path in paths:
        with open(path, "rb") as f:
            rows = pickle.load(f)
        for row in rows:
            row["source"] = os.path.dirname(path)
            if not load_frames:
                row["frame"] = None
            yield row


def load_episodes(*, directory: str, load_frames: bool = True, scene: Optional[str] = None,
                  embed: bool = True) -> list:
    """Episodes as lists of step dicts (step 0 = the reset observation), grouped by (run dir, episode)."""
    episodes: dict = {}
    for row in iter_replay(directory=directory, load_frames=load_frames, scene=scene, embed=embed):
        episodes.setdefault((row["source"], row["episode"]), []).append(row)
    out = []
    for key in sorted(episodes):
        steps = sorted(episodes[key], key=lambda r: r["step"])
        out.append(steps)
    return out
