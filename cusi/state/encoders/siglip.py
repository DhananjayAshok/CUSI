"""
SigLIP 2 embedder: one pooled, L2-normalised vector per canvas frame, zero-shot or fine-tuned.
"""
import os
import torch
from PIL import Image
from cusi.state.encoders.base import ImageEmbedder, check_meta

VISION_WEIGHTS = "vision_model.pt"


class SiglipEmbedder(ImageEmbedder):
    name = "siglip"

    def __init__(self, *, env_name: str, model_name: str, load_path: str = None, device: str = None,
                 batch_size: int = 64) -> None:
        from transformers import AutoModel, AutoProcessor
        super().__init__(env_name=env_name)
        if not model_name:
            raise ValueError("siglip needs --encoder_model (the SigLIP 2 model id)")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        self.load_path = load_path
        if load_path is not None:
            check_meta(directory=load_path, expected={"embedder": self.name, "env": env_name,
                                                      "canvas": list(self.canvas), "base_model": model_name})
            state = torch.load(os.path.join(load_path, VISION_WEIGHTS), map_location="cpu")
            self.model.vision_model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.output_dim = int(self.model.config.vision_config.hidden_size)

    def pil_canvases(self, *, frames: list) -> list:
        out = []
        for f in frames:
            x = self.preprocess(f)
            out.append(Image.fromarray(x[:, :, 0]).convert("RGB") if x.shape[2] == 1 else Image.fromarray(x))
        return out

    def processor_inputs(self, *, frames: list) -> dict:
        inputs = self.processor(images=self.pil_canvases(frames=frames), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        return dict(inputs)

    @staticmethod
    def pooled(*, model, inputs: dict) -> torch.Tensor:
        """The pooled image feature (with grad if the caller wants it: used by fine-tuning)."""
        feats = model.get_image_features(**inputs)
        if not torch.is_tensor(feats):      # some versions return a ModelOutput
            feats = feats.pooler_output
        return feats

    @torch.no_grad()
    def embed(self, *, frames: list) -> torch.Tensor:
        out = []
        for i in range(0, len(frames), self.batch_size):
            inputs = self.processor_inputs(frames=frames[i:i + self.batch_size])
            feats = self.pooled(model=self.model, inputs=inputs)
            out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(out, dim=0)

    def meta(self) -> dict:
        return {**super().meta(), "base_model": self.model_name}
