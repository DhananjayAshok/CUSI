"""`combination`: region_alpha * region + (1 - region_alpha) * embedding (GameBoyRL's
CombinationBuffer with ocr_alpha). A part with weight 0 is neither computed nor stored."""
from cusi.state.scorers.base import Scorer
from cusi.state.scorers.embedding import frame_novelty
from cusi.state.scorers.region import region_novelty


class CombinationScorer(Scorer):
    name = "combination"

    def __init__(self, *, env_name: str, region_alpha: float) -> None:
        if not 0.0 <= region_alpha <= 1.0:
            raise ValueError(f"region_alpha must be in [0, 1], got {region_alpha}")
        self.env_name, self.region_alpha = env_name, region_alpha

    def score(self, *, prev, action, next, archive, add=True, seed_frame=False):
        frame = frame_novelty(next=next, archive=archive, add=add, seed_frame=seed_frame) \
            if self.region_alpha < 1.0 else 0.0
        region = region_novelty(env_name=self.env_name, next=next, archive=archive, add=add) \
            if self.region_alpha > 0.0 else 0.0
        return self.region_alpha * region + (1 - self.region_alpha) * frame, {"frame": frame, "region": region}
