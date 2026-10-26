"""
Legal actions (candidates) for the curiosity policy, per env (plans/archive/next_plan.md §E.2). A candidate is an action type,
an optional target element (an index into the screen's element list) and an optional argument (app or button name),
mapped to the env's own text action, so the env interfaces do not change. There is no typing: like random walks, the
policy acts only on what is on the screen.
"""
import json
from dataclasses import dataclass
from typing import Optional
from cusi.envs.text_boxes import text_boxes_of

MAX_ELEMENTS = 256
#: Action types, shared by every env (their embedding is learned).
TYPES = ("button", "click", "long_press", "scroll_up", "scroll_down", "scroll_left", "scroll_right", "back", "home",
         "open_app", "wait")
#: The apps Android's open_app candidates name (the AndroidWorld apps).
ANDROID_APPS = ("Contacts", "Settings", "Chrome", "Clock", "Camera", "Files", "Phone", "Markor", "Simple Calendar Pro", "Simple SMS Messenger", "Audio Recorder", "Broccoli", "Pro Expense", "OsmAnd", "Retro Music", "Simple Gallery Pro", "Tasks", "VLC", "Joplin", "Simple Draw Pro", "Open Tracks")


@dataclass(frozen=True)
class Candidate:
    type: str
    target: Optional[int]          # index into the element list, or None
    argument: Optional[str]        # app name / button name
    action: str                    # the env's text action


def elements_of(*, obs: dict) -> list:
    """The screen's elements (text boxes, interactive ones without text included), reading order, capped."""
    return text_boxes_of(obs=obs)[:MAX_ELEMENTS]


def gameboy_candidates() -> list:
    # Imported here: GameBoyRL's and WebVoyager's top-level modules cannot share a process.
    from cusi.envs.gameboy import BUTTONS
    return [Candidate(type="button", target=None, argument=b, action=b) for b in BUTTONS]


def android_candidates(*, elements: list) -> list:
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
    for d in ("up", "down", "left", "right"):
        out.append(Candidate(f"scroll_{d}", None, None, json.dumps({"action_type": "scroll", "direction": d})))
    out.append(Candidate("back", None, None, json.dumps({"action_type": "navigate_back"})))
    out.append(Candidate("home", None, None, json.dumps({"action_type": "navigate_home"})))
    out.append(Candidate("wait", None, None, json.dumps({"action_type": "wait"})))
    for app in ANDROID_APPS:
        out.append(Candidate("open_app", None, app, json.dumps({"action_type": "open_app", "app_name": app})))
    return out


def web_candidates(*, elements: list) -> list:
    out = []
    for i, b in enumerate(elements):
        if b.ref is None:
            continue
        out.append(Candidate("click", i, None, f"Thought: explore.\nAction: Click [{b.ref}]"))
        if b.scrollable:
            for d in ("up", "down"):
                out.append(Candidate(f"scroll_{d}", i, None, f"Thought: explore.\nAction: Scroll [{b.ref}]; [{d}]"))
    for d in ("up", "down"):
        out.append(Candidate(f"scroll_{d}", None, None, f"Thought: explore.\nAction: Scroll [WINDOW]; [{d}]"))
    out.append(Candidate("back", None, None, "Thought: explore.\nAction: GoBack"))
    out.append(Candidate("wait", None, None, "Thought: explore.\nAction: Wait"))
    return out


def candidates_for(*, env_name: str, obs: dict) -> tuple:
    """(elements, candidates) for the current observation."""
    if env_name == "gameboy":
        return elements_of(obs=obs), gameboy_candidates()
    elements = elements_of(obs=obs)
    make = android_candidates if env_name == "android" else web_candidates
    return elements, make(elements=elements)
