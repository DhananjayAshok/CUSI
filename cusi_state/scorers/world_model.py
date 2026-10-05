"""`world_model`: 1 - cos(predicted next embedding, real next embedding) (GameBoyRL's WorldModel
curiosity). The model (cusi_explore.world_model architecture: frame stack 2 + action index from
cusi_explore.action_vocab) is loaded from --world_model_load_path, required: a randomly initialised
world model gives a meaningless score. On load it is checked to have been trained with the same
image embedder (world_model_meta.json) and the env's action vocabulary (action_space.json). It is
not trained here, and archive resets don't affect it (the archive is not used).

The action is the env's canonical parsed_action (info["parsed_action"]). A step the env rejected
(next.info["valid"] False) is a no-op (the screen is unchanged) and scores 0, as the archive
scorers give an unchanged screen: world models trained on random-agent replays never see the
"invalid" index, and their arbitrary prediction for it rewarded invalid actions (0.57 mean vs 0.06
for valid steps in the first GameBoy smoke run). The frame before prev is prev.prev_image
(repeated prev frame if missing, as in training).
"""
import torch
from cusi_state.scorers.base import Scorer


class WorldModelScorer(Scorer):
    name = "world_model"

    def __init__(self, *, load_path: str, env_name: str, encoder, device: str = None) -> None:
        from cusi_explore.action_vocab import ActionVocab
        from cusi_explore.world_model import WorldModel, read_meta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = read_meta(directory=load_path)
        if encoder is not None:
            image = encoder.image
            if meta["image_embedder"] != image.name or (meta.get("embedder_meta") not in (None, image.meta())):
                raise ValueError(f"{load_path}: world model trained on {meta['image_embedder']} "
                                 f"({meta.get('embedder_meta')}), this run embeds with {image.name} ({image.meta()})")
        if meta["env"] != env_name:
            raise ValueError(f"{load_path}: world model is for env {meta['env']}, not {env_name}")
        self.vocab = ActionVocab(env_name=env_name)
        spec = ActionVocab.load_spec(directory=load_path)
        if spec["names"] != self.vocab.names:
            raise ValueError(f"{load_path}: action_space.json differs from the current {env_name} vocabulary")
        self.model = WorldModel.load(directory=load_path, device=self.device).eval()

    @torch.no_grad()
    def score(self, *, prev, action, next, archive, add=True, seed_frame=False):
        if prev is None or prev.embedding is None:
            return 0.0, {"world_model": 0.0}
        cur = prev.embedding.image
        before = prev.prev_image if prev.prev_image is not None else cur
        obs = torch.cat([before, cur]).unsqueeze(0).to(self.device)
        if not (next.info or {}).get("valid", True):
            return 0.0, {"world_model": 0.0}
        parsed = action if isinstance(action, dict) else (next.info or {}).get("parsed_action")
        index = self.vocab.encode(parsed_action=parsed)
        pred = self.model(obs=obs, actions=torch.tensor([index], device=self.device))[0]
        real = torch.nn.functional.normalize(next.embedding.image.to(self.device), dim=0)
        value = float(1.0 - (pred * real).sum())
        return value, {"world_model": value}
