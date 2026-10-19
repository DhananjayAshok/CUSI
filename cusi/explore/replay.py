"""
Reading the chunked on-disk replays of exploration runs (written by the removed PPO explorer; search runs are read
by cusi/search/replay.py); step 0 of each episode is the reset observation.
"""
import glob
import os
import pickle
from typing import Iterator, Optional

FIELDS = ("episode", "step", "frame", "embedding", "texts", "generated", "parsed_action", "action_index", "valid",
          "reward_ext", "reward_frame", "reward_text", "reward", "done", "embedder", "reward_region",
          "reward_penalty")


def iter_replay(*, directory: str, load_frames: bool = True, scene: Optional[str] = None) -> Iterator[dict]:
    """Steps of every chunk under `directory` (recursively), in order; a search run directory is read as replay
    (cusi/search/replay.py, one episode per expansion, optionally of one scene)."""
    from cusi.search.replay import is_search_run, iter_search_steps
    if is_search_run(directory=directory):
        yield from iter_search_steps(directory=directory, load_frames=load_frames, scene=scene)
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


def load_episodes(*, directory: str, load_frames: bool = True, scene: Optional[str] = None) -> list:
    """Episodes as lists of step dicts (step 0 = the reset observation), grouped by (run dir, episode)."""
    episodes: dict = {}
    for row in iter_replay(directory=directory, load_frames=load_frames, scene=scene):
        episodes.setdefault((row["source"], row["episode"]), []).append(row)
    out = []
    for key in sorted(episodes):
        steps = sorted(episodes[key], key=lambda r: r["step"])
        out.append(steps)
    return out
