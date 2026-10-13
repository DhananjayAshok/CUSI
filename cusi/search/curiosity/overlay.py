"""
The curiosity session's reward archive (plans/next_plan.md §E.5.1): a fixed base B0 that is only read, plus the current
episode's own states, which are dropped when the episode ends. Steps are scored against B0 ∪ the episode, so the
reward is stationary across a session. It offers the NoveltyArchive interface the scorers use.
"""
from cusi.state.archive import NoveltyArchive


class OverlayRegion:
    def __init__(self, *, base, episode) -> None:
        self.base, self.episode = base, episode

    def score(self, *, boxes: list, prev_boxes=None, add: bool = True) -> float:
        return self.episode.score(boxes=boxes, prev_boxes=prev_boxes, add=add, base=self.base)


class OverlayArchive:
    def __init__(self, *, base: NoveltyArchive) -> None:
        self.base = base
        self.episode = self._empty()

    def _empty(self) -> NoveltyArchive:
        b = self.base
        return NoveltyArchive(metric=b.metric, text_embedder=b.text_embedder, w_image=b.w_image,
                              dedup_threshold=b.dedup_threshold, max_size=10 ** 9,
                              text_seen_threshold=b.text_seen_threshold)

    def new_episode(self) -> None:
        self.episode = self._empty()

    def __len__(self) -> int:
        return len(self.base) + len(self.episode)

    def max_similarity(self, e) -> float:
        best = self.base.max_similarity(e) if len(self.base) else -float("inf")
        if len(self.episode):
            best = max(best, self.episode.max_similarity(e))
        return best

    def novelty(self, e) -> float:
        if len(self) == 0:
            return 1.0
        if self.base.w_image == 1.0:      # image only: GameBoyRL's form, min over both parts
            parts = [p.novelty(e) for p in (self.base, self.episode) if len(p)]
            return min(parts)
        return 1.0 - self.max_similarity(e)

    def add(self, e) -> bool:
        return self.episode.add(e)

    def region(self, name: str):
        return OverlayRegion(base=self.base.region(name), episode=self.episode.region(name))


def local_base(*, like: NoveltyArchive, embedding, obs: dict) -> NoveltyArchive:
    """B0 for --reward_buffer local: empty except the start node (its image and its text keys)."""
    from cusi.envs.text_boxes import text_boxes_of
    from cusi.state.regions import UNSEEN_TEXT
    b = NoveltyArchive(metric=like.metric, text_embedder=like.text_embedder, w_image=like.w_image,
                       dedup_threshold=like.dedup_threshold, max_size=like.max_size,
                       text_seen_threshold=like.text_seen_threshold)
    b.add(embedding)
    b.region(UNSEEN_TEXT).score(boxes=text_boxes_of(obs=obs), add=True)
    return b
