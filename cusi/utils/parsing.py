"""
Parsers for model replies.

Copied from GameBoyRL rather than imported: its top-level `utils` package clashes with WebVoyager's `utils`.
"""
import re
from typing import Optional

PLAN_SEPARATOR = "[STEP]"

# Also accepts a bare "<digits> foo" with no punctuation after the number
_ITEM_RE = re.compile(r"^(?:[-*•]|\d+[.)]?)\s+(.*)$")


_ABSENT = {"NONE", "N/A", "NA"}



def parse_key_value(text: str, key: str) -> Optional[str]:
    """The value after a case-insensitive ``Key:`` (scoped to after a lone "response:"), or None."""
    key_lower = key.lower()
    marker = f"{key_lower}:"

    def clean(value: str) -> Optional[str]:
        value = value.strip()
        stop_idx = value.lower().find("[stop]")
        if stop_idx != -1:
            value = value[:stop_idx]
        value = value.strip()
        return value or None

    text_lower = text.lower()
    if text_lower.count("response:") == 1: # sometimes API models do this. 
        idx = text_lower.index("response:") + len("response:")
        text = text[idx:].strip()
        text_lower = text.lower()

    for line, line_lower in zip(text.splitlines(), text_lower.splitlines()):
        idx = line_lower.find(marker)
        if idx != -1:
            return clean(line[idx + len(marker):])

    if text_lower.count(key_lower) == 1:
        idx = text_lower.index(key_lower)
        rest_of_line = text[idx + len(key):].splitlines()
        return clean(rest_of_line[0]) if rest_of_line else None

    return None



def _is_absent(value: str) -> bool:
    """Whether an item is a "no answer" token or bare punctuation (e.g. markdown residue)."""
    stripped = value.strip()
    return (not stripped
            or stripped.upper() in _ABSENT
            or not any(ch.isalnum() for ch in stripped))


def _heading_index(lines: list[str], marker: str) -> Optional[int]:
    """Index of the line carrying ``marker`` (lowercase, with colon), preferring a heading over a mention."""
    mention = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower().lstrip("*#->•+ \t")
        if stripped.startswith(marker):
            return i
        if mention is None and marker in line.lower():
            mention = i
    return mention


def parse_list(text: str, key: Optional[str] = None) -> list[str]:
    """Bare list items from a reply, optionally scoped to a ``key`` heading (given without the colon)."""
    body = text or ""

    lowered = body.lower()
    if lowered.count("response:") == 1:
        body = body[lowered.index("response:") + len("response:"):]

    lines = body.splitlines()
    items: list[str] = []
    start = 0

    if key is not None:
        index = _heading_index(lines, f"{key.lower()}:")
        if index is None:
            return []
        marker = f"{key.lower()}:"
        line = lines[index]
        # Whatever follows the heading on its own line is the first item.
        head = line[line.lower().index(marker) + len(marker):].strip()
        match = _ITEM_RE.match(head)
        head = match.group(1).strip() if match else head
        if not _is_absent(head):
            items.append(head)
        start = index + 1

    started = bool(items)
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        match = _ITEM_RE.match(stripped)
        if match is None:
            if started:
                break      # next section — stop
            continue       # preamble before the list — skip
        started = True
        value = match.group(1).strip()
        if not _is_absent(value):
            items.append(value)
    return items


def parse_int(text: str, key: str, lo: Optional[int] = None,
              hi: Optional[int] = None) -> Optional[int]:
    """The integer on a ``Key:`` line within inclusive [lo, hi], or None if absent or unparseable."""
    raw = (parse_key_value(text, key) or "").strip()
    if not raw or raw.lower().startswith(("n/a", "na", "none", "unknown")):
        return None

    def in_range(value: int) -> bool:
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    for token in raw.replace(",", " ").split():
        if token.isdigit() and in_range(int(token)):
            return int(token)

    if lo is None and hi is None:
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else None

    for ch in raw:
        if ch.isdigit() and in_range(int(ch)):
            return int(ch)
    return None


def parse_yes_no(text: str, key: str) -> Optional[bool]:
    """True on an explicit yes, False on any other answer, None when the key is absent."""
    raw = parse_key_value(text, key)
    if raw is None:
        return None
    return raw.strip().lower().startswith("yes")



def parse_steps(text: str) -> list:
    """Split a [STEP]-separated plan into steps, dropping empties."""
    if not text:
        return []
    return [part.strip() for part in text.split(PLAN_SEPARATOR) if part.strip()]


def parse_plan(text: str) -> list:
    """The steps of a "Plan:" answer, however the model laid them out."""
    body = text or ""

    lowered = body.lower()
    if lowered.count("response:") == 1:
        body = body[lowered.index("response:") + len("response:"):]

    lines = body.splitlines()
    marker = "plan:"
    index = _heading_index(lines, marker)
    if index is None:
        return []

    head = lines[index]
    rest = head[head.lower().index(marker) + len(marker):]
    block = "\n".join([rest, *lines[index + 1:]]).strip().lstrip("*# \t").strip()

    if not block or block.upper().startswith("NONE"):
        return []
    if PLAN_SEPARATOR in block:
        return parse_steps(block)
    steps = parse_list(text, "Plan")
    if steps:
        return steps
    return [] if _is_absent(block) else [" ".join(block.split())]


def parse_completion(response: str) -> Optional[bool]:
    """The "Complete:" verdict."""
    return parse_yes_no(response, "Complete")

