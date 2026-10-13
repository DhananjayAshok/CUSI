"""
Legal actions (candidates) for the curiosity policy, per env (plans/next_plan.md §E.2). A candidate is an action type,
an optional target element (an index into the screen's element list) and an optional argument (typed string or app
name), mapped to the env's own text action, so the env interfaces do not change.
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Optional
from cusi.envs.text_boxes import text_boxes_of

MAX_ELEMENTS = 256
STRINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "type_strings.json")
#: Action types, shared by every env (their embedding is learned).
TYPES = ("button", "click", "long_press", "scroll_up", "scroll_down", "scroll_left", "scroll_right", "back", "home",
         "open_app", "wait", "type")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z]{2,}")


@dataclass(frozen=True)
class Candidate:
    type: str
    target: Optional[int]          # index into the element list, or None
    argument: Optional[str]        # typed text / app name / button name
    action: str                    # the env's text action


def load_strings() -> dict:
    with open(STRINGS_FILE) as f:
        return json.load(f)


def elements_of(*, obs: dict) -> list:
    """The screen's elements (text boxes, interactive ones without text included), reading order, capped."""
    return text_boxes_of(obs=obs)[:MAX_ELEMENTS]


def type_strings(*, env_name: str, scene: Optional[str], box, strings: dict, max_strings: int) -> list:
    """Up to max_strings strings for one input field: words of its own text, then the scene's and generic lists."""
    own = [w.lower() for w in _WORD_RE.findall(box.text or "")]
    table = strings.get(env_name, {})
    pool = own + table.get("scenes", {}).get(scene or "", []) + table.get("generic", [])
    out = []
    for s in pool:
        if s not in out:
            out.append(s)
    return out[:max_strings]


def gameboy_candidates() -> list:
    # Imported here: GameBoyRL's and WebVoyager's top-level modules cannot share a process.
    from cusi.envs.gameboy import BUTTONS
    return [Candidate(type="button", target=None, argument=b, action=b) for b in BUTTONS]


def android_candidates(*, elements: list, scene: Optional[str], strings: dict, max_type_strings: int) -> list:
    out = []
    for i, b in enumerate(elements):
        if b.ref is None:
            continue
        out.append(Candidate("click", i, None, json.dumps({"action_type": "click", "index": b.ref})))
        if b.interactive:
            out.append(Candidate("long_press", i, None, json.dumps({"action_type": "long_press", "index": b.ref})))
        if b.scrollable:
            for d in ("up", "down", "left", "right"):
                out.append(Candidate(f"scroll_{d}", i, None,
                                     json.dumps({"action_type": "scroll", "direction": d, "index": b.ref})))
        if b.role == "input":
            for s in type_strings(env_name="android", scene=scene, box=b, strings=strings,
                                  max_strings=max_type_strings):
                out.append(Candidate("type", i, s, json.dumps({"action_type": "input_text", "text": s,
                                                               "index": b.ref})))
    for d in ("up", "down", "left", "right"):
        out.append(Candidate(f"scroll_{d}", None, None, json.dumps({"action_type": "scroll", "direction": d})))
    out.append(Candidate("back", None, None, json.dumps({"action_type": "navigate_back"})))
    out.append(Candidate("home", None, None, json.dumps({"action_type": "navigate_home"})))
    out.append(Candidate("wait", None, None, json.dumps({"action_type": "wait"})))
    for app in strings.get("android_apps", []):
        out.append(Candidate("open_app", None, app, json.dumps({"action_type": "open_app", "app_name": app})))
    return out


def web_candidates(*, elements: list, scene: Optional[str], strings: dict, max_type_strings: int) -> list:
    out = []
    for i, b in enumerate(elements):
        if b.ref is None:
            continue
        if b.role == "input":
            for s in type_strings(env_name="web", scene=scene, box=b, strings=strings, max_strings=max_type_strings):
                out.append(Candidate("type", i, s, f"Thought: explore.\nAction: Type [{b.ref}]; {s}"))
        else:
            out.append(Candidate("click", i, None, f"Thought: explore.\nAction: Click [{b.ref}]"))
        if b.scrollable:
            for d in ("up", "down"):
                out.append(Candidate(f"scroll_{d}", i, None, f"Thought: explore.\nAction: Scroll [{b.ref}]; [{d}]"))
    for d in ("up", "down"):
        out.append(Candidate(f"scroll_{d}", None, None, f"Thought: explore.\nAction: Scroll [WINDOW]; [{d}]"))
    out.append(Candidate("back", None, None, "Thought: explore.\nAction: GoBack"))
    out.append(Candidate("wait", None, None, "Thought: explore.\nAction: Wait"))
    return out


def candidates_for(*, env_name: str, obs: dict, scene: Optional[str] = None, strings: Optional[dict] = None,
                   max_type_strings: int = 8) -> tuple:
    """(elements, candidates) for the current observation."""
    if env_name == "gameboy":
        return elements_of(obs=obs), gameboy_candidates()
    strings = strings if strings is not None else load_strings()
    elements = elements_of(obs=obs)
    make = android_candidates if env_name == "android" else web_candidates
    return elements, make(elements=elements, scene=scene, strings=strings, max_type_strings=max_type_strings)
