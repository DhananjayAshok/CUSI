"""
The lookahead model with CUSI's SigLIP world model (cusi.state.scorers.world_model.WorldModel over the fine-tuned
SigLIP's embeddings): the candidate frames are the world model's one-step predictions for every button, decoded by
the SigLIP run's decoder.

Model directory: run_explore.py world_model's output (world_model.pt, world_model_meta.json with embedder_config and
game / game_series, action_space.json, train_summary.json, eval.json). The decoder is the one trained with the SigLIP
fine-tune (embedder_config.embedder_load_path).

Adds to the lookahead record after the step: predicted_similarity (per candidate, cosine of its predicted embedding to
the real next embedding), copy_similarity, chosen_predicted_similarity, hindsight_best, chosen_is_hindsight_best,
reconstruction_pixel_error (the decoder's round trip of the real next frame: the floor), and
base_action_predicted_similarity (the world model's own view of the shadowed base policy's action).
"""
import json
import os
from collections import deque
from typing import Optional
import numpy as np
import torch
from cusi.envs.action_vocab import ActionVocab
from cusi.utils.log_handling import log_info, log_warn
from cusi.world_action.base import read_meta
from cusi.world_action.lookahead import LookaheadModel, grey

FRAME_STACK = 2
_LOADED: dict = {}


class SiglipLookaheadModel(LookaheadModel):
    name = "siglip_lookahead"

    def __init__(self, *, directory: str, meta: dict, embedder, decoder, world_model, vocab: ActionVocab,
                 device: str, parameters: dict) -> None:
        super().__init__(directory=directory, meta=meta, parameters=parameters)
        self.embedder, self.decoder, self.wm, self.vocab, self.device = embedder, decoder, world_model, vocab, device
        self.candidates = [n for n in vocab.names if n != "invalid"]
        self.indices = [vocab.index[n] for n in self.candidates]
        self._stack: deque = deque(maxlen=FRAME_STACK)

    @classmethod
    def load(cls, *, directory: str, device: str, parameters: dict) -> "SiglipLookaheadModel":
        key = (os.path.realpath(directory), device)
        if key in _LOADED:
            return _LOADED[key]
        from cusi.explore.decoder import EmbeddingDecoder
        from cusi.state.encoders import build_image_embedder
        from cusi.state.scorers.world_model import WorldModel
        meta = read_meta(directory=directory)
        config = meta.get("embedder_config")
        if not config or config.get("image_embedder") != "siglip" or not config.get("embedder_load_path"):
            raise ValueError(f"{directory}: siglip_lookahead needs a world model trained on a fine-tuned SigLIP "
                             f"(embedder_config with embedder_load_path); got {config}")
        if meta.get("game") is None:
            raise ValueError(f"{directory}: world_model_meta.json has no 'game'; run tmp_backfill_wm_meta.py or "
                             "retrain so the eval can record what the model was trained on")
        embedder = build_image_embedder(env_name=meta["env"], device=device, parameters=parameters, **config)
        decoder = EmbeddingDecoder.load(directory=config["embedder_load_path"], device=device)
        wm = WorldModel.load(directory=directory, device=device).eval()
        vocab = ActionVocab(env_name=meta["env"])
        spec = ActionVocab.load_spec(directory=directory)
        if spec["names"] != vocab.names:
            raise ValueError(f"{directory}: action_space.json {spec['names']} differs from the current "
                             f"{meta['env']} vocabulary {vocab.names}")
        fitness = {}
        if os.path.exists(os.path.join(directory, "eval.json")):
            e = json.load(open(os.path.join(directory, "eval.json")))
            pc = e.get("pixel_changed", {})
            fitness = {"cos": e.get("cos"), "copy_cos": e.get("copy_cos"), "pixel_changed_predicted": pc.get("predicted"),
                       "pixel_changed_copy": pc.get("copy"), "predicted_beats_copy": pc.get("predicted_beats_copy")}
            if pc and pc.get("predicted", 0) >= pc.get("copy", 0):
                log_warn(f"{directory}: on its held-out changed screens the world model ({pc['predicted']:.1f}) does "
                         f"not beat copying the frame ({pc['copy']:.1f}); planning with it is planning with noise",
                         parameters=parameters)
        else:
            log_warn(f"{directory}: no eval.json; the world model's fitness is unknown", parameters=parameters)
        if os.path.exists(os.path.join(directory, "leak_check.json")):
            fitness["leak_check"] = json.load(open(os.path.join(directory, "leak_check.json")))
        meta = {**meta, "fitness": fitness}
        model = cls(directory=directory, meta=meta, embedder=embedder, decoder=decoder, world_model=wm, vocab=vocab,
                    device=device, parameters=parameters)
        log_info(f"siglip_lookahead: loaded {directory} (game {meta['game']}, series {model.series}, "
                 f"{len(model.candidates)} candidate actions, {device})", parameters=parameters)
        _LOADED[key] = model
        return model

    def summary(self) -> dict:
        return {**super().summary(), "embedder": self.meta["embedder_config"]["embedder_load_path"],
                "fitness": self.meta.get("fitness"), "candidates": self.candidates}

    # ------------------------------------------------------------ state (the frame stack outlives a leg)

    def _embed(self, frame) -> np.ndarray:
        return self.embedder.embed(frames=[frame]).numpy()[0]

    def begin(self, *, task: str, hint: Optional[str], obs: dict, info: dict) -> None:
        super().begin(task=task, hint=hint, obs=obs, info=info)
        if not self._stack:          # a fresh run; a restored stack (an earlier leg's) is kept
            self._stack.append(self._embed(obs["frame"]))

    def history(self):
        return {"stack": [e.copy() for e in self._stack]}

    def restore_history(self, history) -> None:
        self._stack.clear()
        for e in (history or {}).get("stack", []):
            self._stack.append(np.asarray(e, dtype=np.float32))

    # ------------------------------------------------------------ candidates and scores

    @torch.no_grad()
    def candidate_frames(self, *, obs: dict, info: dict) -> tuple:
        # the stack should hold the current screen (begin or observe pushed it); a mismatch means the screen moved
        # without an action we saw, so re-seed from what is on screen now
        cur = self._embed(obs["frame"])
        if not self._stack or float(np.abs(self._stack[-1] - cur).max()) > 1e-4:
            self._stack.append(cur)
        prev = self._stack[-2] if len(self._stack) >= 2 else self._stack[-1]
        latent = torch.tensor(np.concatenate([prev, self._stack[-1]]), device=self.device).unsqueeze(0)
        preds = self.wm(obs=latent.repeat(len(self.indices), 1), actions=torch.tensor(self.indices, device=self.device))
        self._preds = preds.cpu().numpy()
        return self.decoder.decode(preds), {"world_model_action_indices": list(self.indices)}

    @torch.no_grad()
    def observe(self, **kwargs) -> None:
        self._before_emb = self._stack[-1] if self._stack else self._embed(kwargs["obs_before"]["frame"])
        self._after_emb = self._embed(kwargs["obs_after"]["frame"])
        self._stack.append(self._after_emb)          # pushed whether or not the screen changed, as in training
        super().observe(**kwargs)

    @torch.no_grad()
    def score(self, *, pending: dict, obs_before: dict, obs_after: dict) -> None:
        rec, after = pending["record"], grey(obs_after["frame"])
        sims = (self._preds @ self._after_emb).tolist()
        recon = grey(self.decoder.decode(torch.tensor(self._after_emb, device=self.device).unsqueeze(0))[0])
        base_idx = rec.get("base_action_index")
        rec.update({"predicted_similarity": sims, "copy_similarity": float(self._before_emb @ self._after_emb),
                    "chosen_predicted_similarity": sims[pending["chosen"]],
                    "hindsight_best": int(np.argmax(sims)),
                    "chosen_is_hindsight_best": int(np.argmax(sims)) == pending["chosen"],
                    "reconstruction_pixel_error": float(np.abs(recon - after).mean()),
                    "base_action_predicted_similarity": sims[base_idx] if base_idx is not None else None})
