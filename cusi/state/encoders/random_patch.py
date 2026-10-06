"""
Untrained embedder: a fixed random projection of each image patch (GameBoyRL's PatchProjection, any canvas).
"""
import torch
import torch.nn as nn
from cusi.state.encoders.base import ImageEmbedder


def extract_patches(*, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """(N, C, H, W) -> (N, n_patches, C * k * k), patches in row-major (n_h, n_w) order."""
    n, c, h, w = x.shape
    k = kernel_size
    p = x.unfold(2, k, k).unfold(3, k, k)              # (N, C, n_h, n_w, k, k)
    p = p.permute(0, 2, 3, 1, 4, 5).contiguous()       # (N, n_h, n_w, C, k, k)
    return p.view(n, (h // k) * (w // k), c * k * k)


class RandomPatchEmbedder(ImageEmbedder):
    name = "random_patch"

    def __init__(self, *, env_name: str, kernel_size: int = 8, patch_dim: int = 2) -> None:
        super().__init__(env_name=env_name)
        h, w, c = self.canvas
        if h % kernel_size or w % kernel_size:
            raise ValueError(f"canvas {self.canvas} is not divisible into {kernel_size}x{kernel_size} patches")
        self.kernel_size, self.patch_dim = kernel_size, patch_dim
        self.n_patches = (h // kernel_size) * (w // kernel_size)
        self.output_dim = self.n_patches * patch_dim
        # Matches GameBoyRL's make_network; the global RNG is left untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            linear = nn.Linear(kernel_size * kernel_size * c, patch_dim, bias=False)
            nn.init.normal_(linear.weight, mean=0.0, std=1.0)
        self.weight = linear.weight.detach().clone()   # (patch_dim, C*k*k)

    @torch.no_grad()
    def embed(self, *, frames: list) -> torch.Tensor:
        x = self.canvases(frames=frames)
        patches = extract_patches(x=x, kernel_size=self.kernel_size)        # (N, P, C*k*k)
        vec = (patches @ self.weight.T).reshape(x.shape[0], -1)              # (N, P*patch_dim)
        return nn.functional.normalize(vec, dim=-1)

    def meta(self) -> dict:
        return {**super().meta(), "kernel_size": self.kernel_size, "patch_dim": self.patch_dim}
