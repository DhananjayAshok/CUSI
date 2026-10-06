"""
Image and text embedders.
"""
from typing import Any, Optional
from cusi.state.encoders.base import ImageEmbedder, check_meta, write_meta
from cusi.state.encoders.text import TEXT_EMBEDDERS, TextEmbedder, build_text_embedder

IMAGE_EMBEDDERS = ("random_patch", "cnn", "siglip")


def build_image_embedder(*, image_embedder: str, env_name: str, encoder_model: Optional[str] = None,
                         embedder_load_path: Optional[str] = None, device: Optional[str] = None,
                         parameters: dict[str, Any] = None) -> ImageEmbedder:
    if image_embedder == "random_patch":
        from cusi.state.encoders.random_patch import RandomPatchEmbedder
        return RandomPatchEmbedder(env_name=env_name)
    if image_embedder == "siglip":
        from cusi.state.encoders.siglip import SiglipEmbedder
        return SiglipEmbedder(env_name=env_name, model_name=encoder_model, load_path=embedder_load_path,
                              device=device)
    if image_embedder == "cnn":
        from cusi.state.encoders.cnn import CNNEmbedder
        if embedder_load_path is None:
            raise ValueError("--image_embedder cnn needs --embedder_load_path (a trained checkpoint)")
        return CNNEmbedder.load(directory=embedder_load_path, env_name=env_name, device=device)
    raise ValueError(f"--image_embedder must be one of {IMAGE_EMBEDDERS}, got {image_embedder!r}")
