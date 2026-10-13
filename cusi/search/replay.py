"""
A search run read as replay episodes (the format of cusi/explore/replay.py), so the curiosity task source and the
world-model / embedder / decoder training read search runs: each expansion is one episode, its step 0 the expanded
node's screen. Step embeddings are not stored by the search, so they are recomputed with the run's own image
embedder (on frames masked as the search masked them).
"""
import json
import os
from typing import Iterator, Optional
import numpy as np
from cusi.agents.frames import FrameRef

EXPANSIONS_FILE = "expansions.jsonl"


def is_search_run(*, directory: str) -> bool:
    return os.path.exists(os.path.join(directory, EXPANSIONS_FILE))


def _encoder(*, directory: str):
    from cusi.state import build_encoder
    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)
    config = config.get("config", config)
    state = {"image_embedder": config["image_embedder"], "encoder_model": config.get("encoder_model"),
             "embedder_load_path": config.get("embedder_load_path"), "text_embedder": "none",
             "mask_text": config.get("mask_text", True)}
    if state["embedder_load_path"] and not os.path.isabs(state["embedder_load_path"]):
        from cusi.utils.parameter_handling import load_parameters
        state["embedder_load_path"] = os.path.join(load_parameters()["storage_dir"], state["embedder_load_path"])
    return build_encoder(config=state, env_name=config["env"]), config


def iter_search_steps(*, directory: str, load_frames: bool = True, embed: bool = True,
                      scene: Optional[str] = None) -> Iterator[dict]:
    """Replay rows of every committed expansion (optionally of one scene), in order."""
    nodes = {}
    with open(os.path.join(directory, "nodes.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            if "parent" in row:
                nodes[row["id"]] = row
    from cusi.envs.action_vocab import ActionVocab
    encoder, config = _encoder(directory=directory) if embed else (None, {})
    if not config:
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
            config = config.get("config", config)
    vocab = ActionVocab(env_name=config["env"])
    with open(os.path.join(directory, EXPANSIONS_FILE)) as f:
        for line in f:
            exp = json.loads(line)
            start = nodes[exp["node_id"]]
            if scene is not None and start["env_key"] != scene:
                continue
            items = [(start["frame"], start["texts"], start.get("text_boxes") or [], None, {"valid": True}, 0.0, {})]
            items += [(s["frame"], s["texts"], s.get("text_boxes") or [], s["action"], s["info"], s["value"],
                       s["components"]) for s in exp["steps"]]
            frames = [FrameRef(path=os.path.join(directory, it[0])) for it in items]
            embeddings = [None] * len(items)
            if embed:
                embs = encoder.encode(obs_list=[{"frame": fr.numpy(), "text_boxes": it[2]}
                                                for fr, it in zip(frames, items)], info_list=[{}] * len(items))
                embeddings = [np.asarray(e.image, dtype=np.float16) for e in embs]
            for t, ((_f, texts, boxes, action, info, value, comp), frame, emb) in enumerate(zip(items, frames,
                                                                                                 embeddings)):
                yield {"episode": exp["id"], "step": t, "frame": frame if load_frames else None, "embedding": emb,
                       "texts": texts, "text_boxes": boxes, "generated": action,
                       "parsed_action": info.get("parsed_action"),
                       "action_index": None if t == 0 else vocab.encode(parsed_action=info.get("parsed_action")),
                       "valid": bool(info.get("valid", True)), "reward_ext": 0.0,
                       "reward_frame": float(comp.get("frame", 0.0)), "reward_text": float(comp.get("region", 0.0)),
                       "reward_region": float(comp.get("region", 0.0)), "reward": float(value), "reward_penalty": 0.0,
                       "done": t == len(items) - 1, "embedder": config.get("image_embedder"),
                       "scene": start["env_key"], "source": directory, "expander": exp.get("expander")}
