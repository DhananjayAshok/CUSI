"""
Embedding novelty: 1 - max similarity of the next state to the archive.
"""
from cusi.state.scorers.base import Scorer


def frame_novelty(*, next, archive, add: bool, seed_frame: bool) -> float:
    e = next.embedding
    if seed_frame or len(archive) == 0:
        if add:
            archive.add(e)
        return 0.0
    value = archive.novelty(e)
    if add:
        archive.add(e)
    return value


class EmbeddingScorer(Scorer):
    name = "embedding"

    def score(self, *, prev, action, next, archive, add=True, seed_frame=False):
        value = frame_novelty(next=next, archive=archive, add=add, seed_frame=seed_frame)
        return value, {"frame": value}
