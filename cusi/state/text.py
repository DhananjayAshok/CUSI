"""
The text input of the encoders: element lines of every text channel, index numbers stripped.
"""
import re

_INDEX_RE = re.compile(r"(UI element \d+: )|(\"index\": \d+, ?)|(^\[\d+\]:?\s*)|(^\d+[.:)]\s*)")


def element_lines(*, texts: dict) -> list:
    """The element lines of every text channel, with index numbers removed (they shift when one element is added)."""
    lines = []
    for value in (texts or {}).values():
        # WebVoyager joins its elements with tabs on one line ("[1]: <button> "x";\t[2]: ...").
        for line in re.split(r"\n|\t(?=\[\d+\]:)", str(value)):
            line = _INDEX_RE.sub("", line.strip()).strip()
            if line:
                lines.append(line)
    return lines
