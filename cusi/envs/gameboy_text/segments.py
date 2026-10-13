"""
Pixel-level glyph segmentation for games with variable-width fonts (plans/next_plan.md §0, "Per-game risks"): inside
a text region, ink pixels (those differing from the region's paper colour) are split into lines by blank rows and into
glyphs by blank columns; each glyph bitmap is matched exactly against a labelled set. Still template matching, not OCR.

Segment keys are "<height>x<width>:<hex of the row-major bitmap>", the bitmap trimmed to its ink.
"""
import numpy as np

#: A gap of at least this many blank columns between glyphs is a space.
SPACE_GAP = 3
#: Rows whose ink covers more than this share of the region width are box borders, not text.
BORDER_SHARE = 0.8


def segment_key(*, bitmap: np.ndarray) -> str:
    h, w = bitmap.shape
    return f"{h}x{w}:" + np.packbits(bitmap.astype(np.uint8).reshape(-1)).tobytes().hex()


def key_bitmap(*, key: str) -> np.ndarray:
    shape, data = key.split(":")
    h, w = (int(v) for v in shape.split("x"))
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(data), dtype=np.uint8))[:h * w]
    return bits.reshape(h, w).astype(bool)


def ink_mask(*, region: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB -> bool ink: pixels that differ from the most common colour (the paper)."""
    packed = (region[:, :, 0].astype(np.int64) << 16) | (region[:, :, 1].astype(np.int64) << 8) | region[:, :, 2]
    values, counts = np.unique(packed, return_counts=True)
    return packed != values[counts.argmax()]


def _runs(flags: np.ndarray) -> list:
    """(start, end) of consecutive True runs (end exclusive)."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def segment_region(*, region: np.ndarray, min_line_height: int = 5, max_line_height: int = 16) -> list:
    """Lines of glyphs in a text region: [(y0, y1, [(x0, x1, key) or None for a space, ...]), ...] in region
    pixels."""
    ink = ink_mask(region=region)
    rows = ink.mean(axis=1)
    ink[rows > BORDER_SHARE] = False
    # vertical border columns: ink in (almost) every row of the region
    cols = ink.mean(axis=0)
    ink[:, cols > 0.9] = False
    lines = []
    for y0, y1 in _runs(ink.any(axis=1)):
        if not (min_line_height <= y1 - y0 <= max_line_height):
            continue
        band = ink[y0:y1]
        glyphs, last_x1 = [], None
        for x0, x1 in _runs(band.any(axis=0)):
            if last_x1 is not None and x0 - last_x1 >= SPACE_GAP:
                glyphs.append(None)
            bitmap = band[:, x0:x1]
            ys = np.where(bitmap.any(axis=1))[0]
            glyphs.append((x0, x1, segment_key(bitmap=bitmap[ys[0]:ys[-1] + 1]), int(ys[0])))
            last_x1 = x1
        if glyphs:
            lines.append((y0, y1, glyphs))
    return lines


def read_segments(*, screen: np.ndarray, corners: tuple, table: dict) -> tuple:
    """(lines, unknown keys): lines as (text, (x0, y0, x1, y1) in screen pixels) for one region."""
    x0, y0, x1, y1 = corners
    region = np.asarray(screen)[y0:y1, x0:x1, :3]
    glyphs = table.get("segments", {})
    ignore = set(table.get("ignore", []))
    out, unknown = [], []
    for ly0, ly1, items in segment_region(region=region):
        chars, gx0, gx1 = [], None, None
        for item in items:
            if item is None:
                chars.append(" ")
                continue
            sx0, sx1, key, _top = item
            if key in ignore:
                continue
            char = glyphs.get(key)
            if char is None:
                unknown.append(key)
                char = "?"
            chars.append(char)
            gx0 = sx0 if gx0 is None else gx0
            gx1 = sx1
        text = "".join(chars).strip()
        if text.replace("?", "").strip():
            out.append((text, (x0 + gx0, y0 + ly0, x0 + gx1, y0 + ly1)))
    return out, unknown


def segments_at(*, screen: np.ndarray, corners: tuple) -> list:
    """Every glyph segment in a region with a context strip, for harvesting: [(key, strip, (x, w) in strip)]."""
    x0, y0, x1, y1 = corners
    region = np.asarray(screen)[y0:y1, x0:x1, :3]
    out = []
    for ly0, ly1, items in segment_region(region=region):
        for item in items:
            if item is None:
                continue
            sx0, sx1, key, _top = item
            c0, c1 = max(0, sx0 - 16), min(region.shape[1], sx1 + 16)
            out.append((key, region[ly0:ly1, c0:c1].copy(), (sx0 - c0, sx1 - sx0)))
    return out
