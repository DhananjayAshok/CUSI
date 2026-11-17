"""
The learned-world-model preview: oracle_preview_image (cusi.world_action.oracle_preview) with the env's exact
lookahead replaced by a tree world model (cusi.world_model). The base executor proposes; the world model predicts the
next screen from [the previous screen, the current screen] and the proposal (its type and target element box,
cusi.world_model.actions), the prediction is decoded by the fine-tuned SigLIP's decoder and shown to the check as
image 2 (resized to the current screen's size); the check keeps the proposal or sends the executor feedback, as in
oracle_preview_image (images only, every preview checked).

The model reads no env state: only the executor's observation (its screenshot and screen element list, which the
executor itself sees), the env's action parser and its description for the check prompt. A proposal that does not
parse to a world-model action type is kept without a preview.

Model directory: storage/world_model/<env>/<name> (run_world_model.py), or its wm/ subdirectory.
"""
import os
import time
from typing import Callable, Optional
import numpy as np
import torch
from PIL import Image
from cusi.state.canvas import _as_array, frame_for_embedding
from cusi.world_action.base import Plan, game_series
from cusi.world_action.oracle_preview import OraclePreviewImageModel

WM_AFTER = ("Image 2 is a learned world model's prediction of the screen the proposed action leads to: an "
            "approximate, blurred guess, not an exact preview. Judge what it shows, allowing for missing detail.")


class WMPreviewImageModel(OraclePreviewImageModel):
    name = "wm_preview_image"
    one_after = WM_AFTER

    def __init__(self, *, directory: str, meta: dict, embedder, decoder, world_model, device: str,
                 parameters: dict) -> None:
        super().__init__(parameters=parameters)
        self.directory = directory
        self.meta = {"env": meta["env"], "game": meta["game"], "game_series": game_series(game=meta["game"]),
                     "oracle": False, "dataset_dir": meta["dataset_dir"], "embedder_dir": meta["embedder_dir"]}
        self.embedder, self.decoder, self.wm, self.device = embedder, decoder, world_model, device
        self._prev_frame = None
        self._now = None

    @classmethod
    def load(cls, *, directory: str, device: str, parameters: dict) -> "WMPreviewImageModel":
        from cusi.explore.decoder import EmbeddingDecoder
        from cusi.state.encoders.siglip import SiglipEmbedder
        from cusi.world_model.model import BoxActionWorldModel
        from cusi.world_model.train import WM_META, read_wm_meta
        wm_dir = directory if os.path.exists(os.path.join(directory, WM_META)) else os.path.join(directory, "wm")
        meta = read_wm_meta(directory=wm_dir)
        embedder = SiglipEmbedder(env_name=meta["env"], model_name=meta["base_model"], load_path=meta["embedder_dir"],
                                  device=device)
        decoder = EmbeddingDecoder.load(directory=meta["embedder_dir"], device=device).eval()
        wm = BoxActionWorldModel.load(directory=wm_dir, device=device).eval()
        return cls(directory=wm_dir, meta=meta, embedder=embedder, decoder=decoder, world_model=wm, device=device,
                   parameters=parameters)

    def attach_env(self, env) -> None:
        self._env = env      # the parser and the env description only; the game stays the model's own

    def begin(self, *, task: str, hint: Optional[str], obs: dict, info: dict) -> None:
        super().begin(task=task, hint=hint, obs=obs, info=info)
        self._prev_frame = None

    def observe(self, *, action_text: str, obs_before: dict, info_before: dict, obs_after: dict,
                info_after: dict, valid: bool) -> None:
        self._prev_frame = frame_for_embedding(obs=obs_before, info=info_before)

    def plan(self, *, obs: dict, info: dict, error: Optional[str], call: Callable,
             policy: Optional[Callable] = None) -> Plan:
        self._now = (obs, info)
        return super().plan(obs=obs, info=info, error=error, call=call, policy=policy)

    @torch.no_grad()
    def _preview(self, *, action: str) -> dict:
        from cusi.search.tree import boxes_as_dicts
        from cusi.world_model.actions import action_box, action_type, parse_executor_action
        t0 = time.time()
        obs, info = self._now
        parsed = parse_executor_action(env_name=self.env_name, env=self._env, text=action)
        kind = action_type(env_name=self.env_name, parsed=parsed)
        if kind is None:
            return {"failed": "the action has no world-model action type", "seconds": time.time() - t0}
        cur = _as_array(frame_for_embedding(obs=obs, info=info))
        prev = cur if self._prev_frame is None else self._prev_frame
        emb = self.embedder.embed(frames=[prev, cur]).to(self.device)
        box = action_box(env_name=self.env_name, parsed=parsed, text_boxes=boxes_as_dicts(obs.get("text_boxes")),
                         frame_size=(cur.shape[1], cur.shape[0]))
        pred = self.wm(obs=emb.reshape(1, -1), types=torch.tensor([self.wm.type_index[kind]], device=self.device),
                       boxes=torch.tensor([box], device=self.device))
        decoded = self.decoder.decode(pred)[0]
        frame = np.asarray(Image.fromarray(decoded).resize((cur.shape[1], cur.shape[0]), Image.BILINEAR))
        return {"frame": frame, "frames": [frame], "text": "", "valid": True, "error": None, "changed": True,
                "true": {}, "synced": None, "agent_state": None, "seconds": time.time() - t0,
                "wm_type": kind, "wm_box": box}
