"""
Trained embedder: GameBoyRL's per-patch conv autoencoder, generalised to any canvas.
"""
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from cusi.state.encoders.base import ImageEmbedder, check_meta, write_meta
from cusi.state.encoders.random_patch import extract_patches

WEIGHTS = "embedder.pt"


def _init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class PatchAutoencoder(nn.Module):
    def __init__(self, *, channels: int, kernel_size: int = 8, patch_dim: int = 4) -> None:
        super().__init__()
        self.channels, self.kernel_size, self.patch_dim = channels, kernel_size, patch_dim
        w1 = 2 * channels
        self.patch_norm = nn.BatchNorm2d(channels, affine=False)
        convs = nn.Sequential(_init(nn.Conv2d(channels, w1, 1)), nn.ReLU(), _init(nn.Conv2d(w1, 4, 2)), nn.ReLU(),
                              _init(nn.Conv2d(4, 4, 2)), nn.ReLU())
        with torch.no_grad():
            out = convs(torch.zeros(1, channels, kernel_size, kernel_size))
        self.chain_shape = tuple(out.shape[1:])
        chain_dim = int(np.prod(self.chain_shape))
        self.encoder = nn.Sequential(*convs, nn.Flatten(), _init(nn.Linear(chain_dim, patch_dim)), nn.Sigmoid(),
                                     nn.LayerNorm(patch_dim))
        self.decoder_linear = _init(nn.Linear(patch_dim, chain_dim))
        self.decoder = nn.Sequential(_init(nn.ConvTranspose2d(4, 4, 2)), nn.ReLU(), _init(nn.ConvTranspose2d(4, w1, 2)),
                                     nn.ReLU(), _init(nn.ConvTranspose2d(w1, channels, 1)))

    def encode_patches(self, patches: torch.Tensor) -> tuple:
        """patches (N, P, C*k*k) in [0, 255] -> (unit patch vectors, BatchNorm-normed patches)."""
        n, p, _ = patches.shape
        x = self.patch_norm(patches.reshape(n * p, self.channels, self.kernel_size, self.kernel_size))
        z = nn.functional.normalize(self.encoder(x), dim=-1)
        return z.view(n, p, self.patch_dim), x

    def decode_patches(self, z: torch.Tensor) -> torch.Tensor:
        """(N, P, patch_dim) -> normed-space patches (N*P, C, k, k)."""
        n, p, _ = z.shape
        return self.decoder(self.decoder_linear(z.reshape(n * p, -1)).view(-1, *self.chain_shape))

    def forward(self, patches: torch.Tensor) -> tuple:
        """Training: (reconstruction, target), both in the BatchNorm-normalised patch space."""
        z, target = self.encode_patches(patches)
        return self.decode_patches(z), target


class CNNEmbedder(ImageEmbedder):
    name = "cnn"

    def __init__(self, *, env_name: str, kernel_size: int = 8, patch_dim: int = 4, device: Optional[str] = None,
                 net: Optional[PatchAutoencoder] = None) -> None:
        super().__init__(env_name=env_name)
        h, w, c = self.canvas
        self.kernel_size, self.patch_dim = kernel_size, patch_dim
        self.n_patches = (h // kernel_size) * (w // kernel_size)
        self.output_dim = self.n_patches * patch_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = (net or PatchAutoencoder(channels=c, kernel_size=kernel_size, patch_dim=patch_dim)).to(self.device)

    def patches(self, *, frames: list) -> torch.Tensor:
        return extract_patches(x=self.canvases(frames=frames), kernel_size=self.kernel_size)

    @torch.no_grad()
    def embed(self, *, frames: list, batch_size: int = 256) -> torch.Tensor:
        self.net.eval()
        out = []
        for i in range(0, len(frames), batch_size):
            z, _ = self.net.encode_patches(self.patches(frames=frames[i:i + batch_size]).to(self.device))
            out.append(nn.functional.normalize(z.reshape(z.shape[0], -1), dim=-1).cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def decode(self, embedding: torch.Tensor) -> np.ndarray:
        """(N, output_dim) -> canvases (N, H, W, C) uint8 (patch vectors renormalised, BatchNorm undone)."""
        self.net.eval()
        h, w, c = self.canvas
        k = self.kernel_size
        z = nn.functional.normalize(embedding.to(self.device).view(-1, self.n_patches, self.patch_dim), dim=-1)
        x = self.net.decode_patches(z)                                               # (N*P, C, k, k)
        bn = self.net.patch_norm
        x = x * (bn.running_var.view(1, -1, 1, 1) + bn.eps).sqrt() + bn.running_mean.view(1, -1, 1, 1)
        n = z.shape[0]
        x = x.view(n, h // k, w // k, c, k, k).permute(0, 1, 4, 2, 5, 3).reshape(n, h, w, c)
        return x.clamp(0, 255).round().byte().cpu().numpy()

    def meta(self) -> dict:
        return {**super().meta(), "kernel_size": self.kernel_size, "patch_dim": self.patch_dim}

    def save(self, *, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "kernel_size": self.kernel_size, "patch_dim": self.patch_dim},
                   os.path.join(directory, WEIGHTS))
        write_meta(directory=directory, meta=self.meta())

    @classmethod
    def load(cls, *, directory: str, env_name: str, device: Optional[str] = None) -> "CNNEmbedder":
        from cusi.state.canvas import canvas_shape
        check_meta(directory=directory, expected={"embedder": cls.name, "env": env_name,
                                                  "canvas": list(canvas_shape(env_name=env_name))})
        ckpt = torch.load(os.path.join(directory, WEIGHTS), map_location="cpu")
        emb = cls(env_name=env_name, kernel_size=ckpt["kernel_size"], patch_dim=ckpt["patch_dim"], device=device)
        emb.net.load_state_dict(ckpt["state_dict"])
        emb.net.eval()
        return emb
