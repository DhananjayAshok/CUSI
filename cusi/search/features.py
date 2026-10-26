"""
Node features (plans/archive/next_plan.md §A): once per new node, a VLM lists the interesting features or functionalities that
exploring from its screen could surface. The decision scorer (§B) later checks them against what has been discovered.
"""
from cusi.envs.text_boxes import render_boxes
from cusi.search.describe import parse_bullets

MAX_TEXT_LINES = 80
MAX_TEXT_CHARS = 4000

FEATURES_PROMPT = """You are looking at a screen of {env_description}. Someone is exploring this environment with no \
particular task, trying to find as many different screens, menus, settings, content, interactions or game areas as \
possible. List the interesting features or functionalities that exploring from this exact screen could surface: \
things reachable from here that such an explorer would want to find.

Text currently on the screen:
{texts}

Reply in exactly this format, with at most {max_features} features, each a short noun phrase:
Features:
- <feature>
- <feature>"""


def screen_text(*, boxes: list) -> str:
    """The screen's text boxes ('<role>: <text>' lines), truncated for prompts."""
    out = render_boxes(boxes=boxes, max_lines=MAX_TEXT_LINES, max_chars=MAX_TEXT_CHARS)
    return out if out else "(no text)"


def parse_features(*, text: str, max_features: int) -> tuple:
    """(features, parsed): the bullets after 'Features:' (or anywhere, if the label is missing)."""
    text = str(text or "")
    lower = text.lower()
    block = text[lower.index("features:") + len("features:"):] if "features:" in lower else text
    features = parse_bullets(block=block, limit=max_features)
    return features, bool(features)


class VLMFeatures:
    def __init__(self, *, vlm, max_features: int = 8, max_new_tokens: int = 256, temperature: float = 0.0) -> None:
        self.vlm, self.max_features = vlm, max_features
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.n_calls = self.n_failed = 0

    def prompt(self, *, env_description: str, text_boxes: list) -> str:
        return FEATURES_PROMPT.format(env_description=env_description, texts=screen_text(boxes=text_boxes),
                                      max_features=self.max_features)

    def __call__(self, *, env_description: str, frame, text_boxes: list) -> dict:
        """{"features", "features_raw", "parsed"} (the first two are the node's fields); frame is the node's
        unlabelled screen."""
        out = str(self.vlm.infer(texts=self.prompt(env_description=env_description, text_boxes=text_boxes),
                                 images=[frame], max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature)["output"])
        self.n_calls += 1
        features, parsed = parse_features(text=out, max_features=self.max_features)
        self.n_failed += not parsed
        return {"features": features, "features_raw": out, "parsed": parsed}
