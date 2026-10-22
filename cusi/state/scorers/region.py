"""
Region novelty: the share of the screen's text boxes never seen (UnseenTextBuffer), on every env.
"""
from cusi.envs.text_boxes import text_boxes_of
from cusi.state.regions import UNSEEN_TEXT
from cusi.state.scorers.base import Scorer


def region_novelty(*, prev, next, archive, add: bool) -> float:
    return archive.region(UNSEEN_TEXT).score(boxes=text_boxes_of(obs=next.obs),
                                             prev_boxes=text_boxes_of(obs=prev.obs) if prev is not None else [], add=add)


def remember_text(*, next, archive) -> None:
    archive.region(UNSEEN_TEXT).add(boxes=text_boxes_of(obs=next.obs))


class RegionScorer(Scorer):
    name = "region"

    def score(self, *, prev, action, next, archive, add=True, seed_frame=False):
        value = region_novelty(prev=prev, next=next, archive=archive, add=add)
        return value, {"region": value}

    def remember(self, *, next, archive):
        remember_text(next=next, archive=archive)
