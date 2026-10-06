"""
Per-environment discrete action vocabularies for the world model.
Element indices >= K share one overflow slot; typed text is not encoded; "invalid" is a no-op.
"""
import json
import os
from typing import Optional

K_ANDROID = 64
K_WEB = 128
GAMEBOY_BUTTONS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START")
DIRECTIONS4 = ("up", "down", "left", "right")
ACTION_SPACE_FILENAME = "action_space.json"


def _indexed(prefix: str, k: int) -> list:
    return [f"{prefix}[{i}]" for i in range(k)] + [f"{prefix}[>={k}]"]


def vocabulary(*, env_name: str) -> list:
    if env_name == "gameboy":
        names = list(GAMEBOY_BUTTONS)
    elif env_name == "android":
        names = []
        for t in ("click", "long_press", "input_text"):
            names += _indexed(t, K_ANDROID)
        for d in DIRECTIONS4:
            names += _indexed(f"scroll_{d}", K_ANDROID)
        names += [f"scroll_{d}" for d in DIRECTIONS4]
        names += ["navigate_back", "navigate_home", "keyboard_enter", "wait", "open_app", "answer", "status"]
    elif env_name == "web":
        names = _indexed("click", K_WEB) + _indexed("type", K_WEB)
        for d in ("up", "down"):
            names += _indexed(f"scroll_{d}", K_WEB)
        names += ["scroll_window_up", "scroll_window_down", "wait", "goback", "google", "answer"]
    else:
        raise ValueError(f"unknown env {env_name}")
    return names + ["invalid"]


class ActionVocab:
    def __init__(self, *, env_name: str) -> None:
        self.env_name = env_name
        self.names = vocabulary(env_name=env_name)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.k = {"android": K_ANDROID, "web": K_WEB}.get(env_name)

    def __len__(self) -> int:
        return len(self.names)

    def _slot(self, prefix: str, element: int) -> int:
        element = int(element)
        return self.index[f"{prefix}[{element}]" if element < self.k else f"{prefix}[>={self.k}]"]

    def encode(self, *, parsed_action: Optional[dict]) -> int:
        """Index of a canonical action (info["parsed_action"]); "invalid" for None/unknown."""
        if not parsed_action:
            return self.index["invalid"]
        a = parsed_action
        t = a.get("action_type")
        try:
            if self.env_name == "gameboy":
                return self.index.get(str(a.get("low_level_action", t)).replace("PRESS_ARROW_", "")
                                      .replace("PRESS_BUTTON_", ""), self.index["invalid"])
            if self.env_name == "android":
                if t in ("click", "long_press", "input_text") and a.get("index") is not None:
                    return self._slot(t, a["index"])
                if t == "scroll":
                    d = a.get("direction", "down")
                    return self._slot(f"scroll_{d}", a["index"]) if a.get("index") is not None \
                        else self.index[f"scroll_{d}"]
                return self.index.get(t, self.index["invalid"])
            if t in ("click", "type"):
                return self._slot(t, a["index"])
            if t == "scroll":
                d = a.get("direction", "down")
                return self.index[f"scroll_window_{d}"] if a.get("target") == "WINDOW" \
                    else self._slot(f"scroll_{d}", a["target"])
            return self.index.get(t, self.index["invalid"])
        except (KeyError, ValueError, TypeError):
            return self.index["invalid"]

    def save(self, *, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, ACTION_SPACE_FILENAME), "w") as f:
            json.dump({"env": self.env_name, "n_actions": len(self), "k": self.k, "names": self.names}, f, indent=1)

    @staticmethod
    def load_spec(*, directory: str) -> dict:
        with open(os.path.join(directory, ACTION_SPACE_FILENAME)) as f:
            return json.load(f)
