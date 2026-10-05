"""The text input of the encoders (provisional, curiosity_plan §3.2.1 / §4): the element lines of
every obs["texts"] channel, index numbers stripped (one extra element renumbers every later
line), bounding boxes ignored. GameBoy has no text channel yet ([GB-OCR]): no lines.
"""
import re

_INDEX_RE = re.compile(r"(UI element \d+: )|(\"index\": \d+, ?)|(^\[\d+\]:?\s*)|(^\d+[.:)]\s*)")


def element_lines(*, texts: dict) -> list:
    """The element lines of every text channel, with index numbers removed."""
    lines = []
    for value in (texts or {}).values():
        # WebVoyager joins its elements with tabs on one line ("[1]: <button> "x";\t[2]: ...") (D2.16).
        for line in re.split(r"\n|\t(?=\[\d+\]:)", str(value)):
            line = _INDEX_RE.sub("", line.strip()).strip()
            if line:
                lines.append(line)
    return lines
