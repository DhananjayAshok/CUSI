"""
VLM prior P(n): a VLM rates how promising a new node is to explore from, once per node.
"""
import re
from typing import Optional
from cusi.search.expanders import flatten_texts

PRIOR_PROMPT = """You are looking at a screen of {env_description}. An explorer wants to discover as \
many different screens, features and kinds of content of this environment as possible, with no \
particular task. How promising is it to keep exploring from this exact screen? Consider whether \
new, unseen things are likely reachable from here in a few actions.

Text currently on the screen:
{texts}

Answer in exactly this format and nothing else:
Score: <an integer from 0 to 10>
Reason: <one short line>"""

_SCORE_RE = re.compile(r"score\s*[:=]\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*[:=]\s*\**\s*(.+)", re.IGNORECASE)


def parse_prior(*, text: str) -> tuple:
    """(score in [0, 10] or None, reason)."""
    m = _SCORE_RE.search(text or "")
    score = None
    if m:
        score = min(10.0, max(0.0, float(m.group(1))))
    r = _REASON_RE.search(text or "")
    return score, (r.group(1).strip().splitlines()[0][:300] if r else "")


# TODO: revisit the prior. It sees only this node's frame and texts, not what the archive or tree already
# holds, so it scores in absolute terms; and it is computed once at creation, so it goes stale as the search
# grows. Options: show the k nearest existing nodes' frames or a list of explored screens; recompute on
# selection. See plans/tree_search_plan.md §7.
class VLMPrior:
    def __init__(self, *, vlm, max_new_tokens: int = 96, temperature: float = 0.0, fallback: float = 0.5) -> None:
        self.vlm, self.max_new_tokens, self.temperature, self.fallback = vlm, max_new_tokens, temperature, fallback
        self.n_calls = 0
        self.n_failed = 0

    def __call__(self, *, env_description: str, frame, texts: dict) -> dict:
        """{"prior", "reason", "raw", "parsed"}."""
        text = PRIOR_PROMPT.format(env_description=env_description, texts=flatten_texts(texts=texts))
        out = str(self.vlm.infer(texts=text, images=[frame], max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature)["output"])
        self.n_calls += 1
        score, reason = parse_prior(text=out)
        if score is None:
            self.n_failed += 1
            return {"prior": self.fallback, "reason": reason, "raw": out, "parsed": False}
        return {"prior": score / 10.0, "reason": reason, "raw": out, "parsed": True}
