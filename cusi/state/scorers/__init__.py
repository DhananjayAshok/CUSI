"""
Novelty scorers.
"""
from typing import Any, Optional
from cusi.state.scorers.base import Scorer

NOVELTY_SCORERS = ("embedding", "region", "combination", "world_model")


def build_scorer(*, novelty_scorer: str, env_name: str, region_alpha: float = 0.5,
                 world_model_load_path: Optional[str] = None, encoder=None, device: Optional[str] = None) -> Scorer:
    if novelty_scorer == "embedding":
        from cusi.state.scorers.embedding import EmbeddingScorer
        return EmbeddingScorer()
    if novelty_scorer == "region":
        from cusi.state.scorers.region import RegionScorer
        return RegionScorer(env_name=env_name)
    if novelty_scorer == "combination":
        from cusi.state.scorers.combination import CombinationScorer
        return CombinationScorer(env_name=env_name, region_alpha=region_alpha)
    if novelty_scorer == "world_model":
        from cusi.state.scorers.world_model import WorldModelScorer
        if world_model_load_path is None:
            raise ValueError("--novelty_scorer world_model needs --world_model_load_path")
        return WorldModelScorer(load_path=world_model_load_path, env_name=env_name, encoder=encoder, device=device)
    raise ValueError(f"--novelty_scorer must be one of {NOVELTY_SCORERS}, got {novelty_scorer!r}")
