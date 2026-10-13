"""
The unified text of a screen: TextBox items (text, box in raw-frame pixels, role, interactive, source), in reading
order. Every env puts them in obs["text_boxes"]; obs["texts"] keeps the env's labelled lists for the action layer.
"""
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

ROLES = ("heading", "body", "label", "button", "link", "input", "menu", "dialogue", "text")
SOURCES = ("a11y", "dom", "tiles")
#: Rows whose top edges are within this many pixels count as one line in the reading order.
LINE_TOLERANCE = 8


@dataclass
class TextBox:
    text: str                      # cleaned, whitespace-normalised ("" for interactive elements without text)
    box: tuple                     # (x0, y0, x1, y1) in pixels of the unlabelled frame
    role: str                      # one of ROLES
    interactive: bool
    source: str                    # one of SOURCES
    ref: Optional[int] = None      # the env's element index for actions (Android ui_elements / Web set-of-mark)
    scrollable: bool = False

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = [int(round(v)) for v in self.box]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TextBox":
        return cls(text=d["text"], box=tuple(d["box"]), role=d["role"], interactive=bool(d["interactive"]),
                   source=d["source"], ref=d.get("ref"), scrollable=bool(d.get("scrollable", False)))


def clean_text(*, text: Optional[str]) -> str:
    return " ".join(str(text or "").split())


def as_boxes(*, boxes: Iterable) -> list:
    """TextBox items from TextBox or dict items (stored rows)."""
    return [b if isinstance(b, TextBox) else TextBox.from_dict(b) for b in (boxes or [])]


def reading_order(*, boxes: list) -> list:
    """Top to bottom, then left to right; tops within LINE_TOLERANCE are one line."""
    out, line, line_top = [], [], None
    for b in sorted(boxes, key=lambda b: (b.box[1], b.box[0])):
        if line_top is not None and b.box[1] - line_top > LINE_TOLERANCE:
            out += sorted(line, key=lambda b: b.box[0])
            line = []
        if not line:
            line_top = b.box[1]
        line.append(b)
    return out + sorted(line, key=lambda b: b.box[0])


def clip_box(*, box: tuple, width: int, height: int) -> Optional[tuple]:
    """box clipped to the frame, or None if nothing of it is on screen."""
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def contains(*, outer: tuple, inner: tuple) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def render_boxes(*, boxes: list, max_lines: Optional[int] = None, max_chars: Optional[int] = None) -> str:
    """'<role>: <text>' per box with text, in order, for prompts."""
    lines = [f"{b.role}: {b.text}" for b in as_boxes(boxes=boxes) if b.text]
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"({len(lines) - max_lines} more)"]
    out = "\n".join(lines)
    return out[:max_chars] if max_chars is not None else out


def text_boxes_of(*, obs: Optional[dict]) -> list:
    """obs["text_boxes"] as TextBox items ([] if absent)."""
    return as_boxes(boxes=(obs or {}).get("text_boxes"))


ROLE_COLOURS = {"heading": (220, 40, 40), "body": (40, 120, 220), "label": (40, 170, 70), "button": (230, 140, 0),
                "link": (150, 60, 200), "input": (0, 170, 170), "menu": (200, 0, 140), "dialogue": (220, 40, 40),
                "text": (90, 90, 90)}


def overlay(*, frame, boxes: list, scale: int = 1, label: bool = True):
    """A PIL image of frame with each box outlined in its role's colour and its text written above it (checks)."""
    import numpy as np
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for b in as_boxes(boxes=boxes):
        x0, y0, x1, y1 = (v * scale for v in b.box)
        colour = ROLE_COLOURS.get(b.role, (0, 0, 0))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=colour, width=2 if b.interactive else 1)
        if label:
            text = (b.text or "<no text>")[:60]
            draw.text((x0 + 1, max(0, y0 - 10)), text, fill=colour)
    return img
