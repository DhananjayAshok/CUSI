"""
Actions for the tree world models: a discrete action type plus the box of the element it acts on.

An indexed action (Web Click / Type / Scroll [n], Android click / long_press / input_text / scroll with an index)
names an element by its number on the screen it was taken from; the frame the model sees is the unlabelled
screenshot, so the number alone is not grounded. Its box is that element's box on the screen before the action
(the screen's text_boxes entry whose "ref" is the index), as [x0, y0, x1, y1] divided by the frame's width and height.
Typing is one type whatever is typed (the text is not encoded); an Android open_app is one type whatever the app.
Actions without an element (GameBoy buttons, Scroll [WINDOW], back, wait, ...) have no box.
"""
from typing import Optional
from cusi.envs.action_vocab import GAMEBOY_BUTTONS

ACTION_TYPES = {
    "gameboy": list(GAMEBOY_BUTTONS),
    "web": ["click", "type", "scroll_up", "scroll_down", "scroll_window_up", "scroll_window_down", "wait", "goback",
            "answer"],
    "android": ["click", "long_press", "input_text", "scroll_up", "scroll_down", "scroll_left", "scroll_right",
                "navigate_back", "navigate_home", "keyboard_enter", "wait", "open_app", "answer", "status"],
}
#: box features: x0, y0, x1, y1 (fractions of the frame) and 1 if the action has a box
BOX_DIM = 5
NO_BOX = [0.0, 0.0, 0.0, 0.0, 0.0]


def action_types(*, env_name: str) -> list:
    if env_name not in ACTION_TYPES:
        raise ValueError(f"no action types for env {env_name!r}")
    return ACTION_TYPES[env_name]


def action_type(*, env_name: str, parsed: Optional[dict]) -> Optional[str]:
    """The type of a canonical action (info["parsed_action"]); None if it has none in this env."""
    if not parsed:
        return None
    t = parsed.get("action_type")
    if env_name == "gameboy":
        name = str(parsed.get("low_level_action", t)).replace("PRESS_ARROW_", "").replace("PRESS_BUTTON_", "")
        return name if name in GAMEBOY_BUTTONS else None
    if t == "scroll":
        d = parsed.get("direction", "down")
        if env_name == "web":
            t = f"scroll_window_{d}" if parsed.get("target") == "WINDOW" else f"scroll_{d}"
        else:
            t = f"scroll_{d}"
    return t if t in ACTION_TYPES[env_name] else None


def target_index(*, env_name: str, parsed: dict) -> Optional[int]:
    """The element number an indexed action names, if any."""
    if env_name == "web":
        index = parsed.get("index", parsed.get("target"))
    elif env_name == "android":
        index = parsed.get("index")
    else:
        return None
    try:
        return None if index is None or index == "WINDOW" else int(index)
    except (TypeError, ValueError):
        return None


def action_box(*, env_name: str, parsed: dict, text_boxes: Optional[list], frame_size: tuple) -> list:
    """BOX_DIM features: the target element's box on the screen before the action, as fractions of (width, height),
    and 1; an Android action given by coordinates (x, y) gets that point as its box; NO_BOX otherwise."""
    width, height = frame_size
    box = None
    index = target_index(env_name=env_name, parsed=parsed)
    if index is not None:
        for b in text_boxes or []:
            if b.get("ref") == index and b.get("box") is not None:
                box = b["box"]
                break
    elif env_name == "android" and parsed.get("x") is not None and parsed.get("y") is not None:
        box = [parsed["x"], parsed["y"], parsed["x"], parsed["y"]]
    if box is None:
        return list(NO_BOX)
    x0, y0, x1, y1 = (float(v) for v in box)
    clip = lambda v: min(1.0, max(0.0, v))
    return [clip(x0 / width), clip(y0 / height), clip(x1 / width), clip(y1 / height), 1.0]


def parse_executor_action(*, env_name: str, env, text: str) -> Optional[dict]:
    """The canonical action an executor's reply would be executed as (what the env's step would record as
    parsed_action), with the env's own parser; None if it does not parse. Reads no env state."""
    if env_name == "web":
        from cusi.envs.webvoyager import action_text, canonical_action
        try:
            key, info = env._wv.extract_information(action_text(text=text))
            return canonical_action(action_key=key, info=info) if key else None
        except Exception:
            return None
    if env_name == "android":
        from cusi.envs.android_world import parse_action
        parsed, _error = parse_action(text=text)
        return parsed
    from cusi.envs.gameboy import canonical_action
    action_class, kwargs = env._parse(text=text)
    return None if action_class is None else canonical_action(action_class=action_class, kwargs=kwargs)
