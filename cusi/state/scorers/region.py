"""
Region novelty: OCR regions on GameBoy, new element lines on Android / Web.
"""
from cusi.state.regions import OCR, TEXT_LINES
from cusi.state.scorers.base import Scorer


def region_novelty(*, env_name: str, next, archive, add: bool) -> float:
    if env_name == "gameboy":
        return archive.region(OCR).score(text_regions=(next.info or {}).get("text_regions"), add=add)
    return archive.region(TEXT_LINES).score(texts=(next.obs or {}).get("texts", {}), add=add)


class RegionScorer(Scorer):
    name = "region"

    def __init__(self, *, env_name: str) -> None:
        self.env_name = env_name

    def score(self, *, prev, action, next, archive, add=True, seed_frame=False):
        value = region_novelty(env_name=self.env_name, next=next, archive=archive, add=add)
        return value, {"region": value}
