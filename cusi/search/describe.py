"""
Transition descriptions: after an expansion, a VLM sees its whole sub-trajectory (the start screen, then the screen
after every action) and describes what happened, listing the features it surfaced (`Discovered:`).

Long trajectories are cut into windows that fit the server's limits (vllm_max_images, vllm_max_model_len): every window
is described in one batched call, then a text-only call composes the window descriptions (GameBoyRL's
window_trajectory). Windows overlap by one frame, so an action's before and after screens share a window.
"""
import math
import re
from typing import Any, Optional
from PIL import Image
from cusi.search.context import action_line
from cusi.utils.parameter_handling import load_parameters

#: Tokens kept free for the prompt text in an image call.
PROMPT_RESERVE = 2000
#: The Qwen-VL patch: one image token per PATCH x PATCH pixels (an estimate for other model families).
PATCH = 28

DESCRIBE_PROMPT = """These {n_images} screenshots show {env_description}, in order: the starting screen, then the \
screen after each of the {n_actions} actions below.

Actions, in order:
{actions}

Describe in two or three sentences what happens over this sequence: where it starts, what the actions did \
(including any that had no visible effect), and where it ends. Name the screens, menus, pages, places or features \
reached. Then list the features or functionalities of the environment that this sequence actually surfaced (screens, \
menus, settings, content, interactions or game areas that were reached or shown), as short noun phrases.

Reply in exactly this format:
Description: <two or three sentences>
Discovered:
- <feature>
- <feature>"""

WINDOW_PROMPT = """These {n_images} screenshots are frames {first}-{last} of a sequence of {n_frames} frames showing \
{env_description}. Frame 1 of the sequence is the starting screen; frame k+1 is the screen after action k. These \
frames are separated by actions {first_action}-{last_action}:
{actions}

Describe in two or three sentences what happens in these frames: what the actions did (including any that had no \
visible effect) and which screens, menus, pages, places or features were reached. Cite frame numbers. Reply with the \
description only."""

COMPOSE_PROMPT = """A sequence of {n_frames} screenshots of {env_description} was described in parts, each covering a \
range of frames (frame 1 is the starting screen; frame k+1 is the screen after action k).

The actions, in order:
{actions}

The descriptions of the parts:
{windows}

Write one description of the whole sequence in two or three sentences: where it starts, what the actions did, and \
where it ends, citing frame ranges. Then list the features or functionalities of the environment that the sequence \
actually surfaced (screens, menus, settings, content, interactions or game areas that were reached or shown), as \
short noun phrases.

Reply in exactly this format:
Description: <two or three sentences>
Discovered:
- <feature>
- <feature>"""

_DESC_RE = re.compile(r"description\s*:\s*(.*?)(?=\n\s*discovered\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_DISC_RE = re.compile(r"discovered\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def normalise_phrase(*, text: str) -> str:
    return " ".join(str(text).casefold().split())


def parse_bullets(*, block: str, limit: Optional[int] = None, max_chars: int = 200) -> list:
    """'- item' lines (also '* item', '1. item'), stripped, without empties or (normalised) duplicates."""
    out, seen = [], set()
    for line in (block or "").splitlines():
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("*").strip()
        if not item or not re.match(r"^\s*(?:[-*•]|\d+[.)])", line):
            continue
        key = normalise_phrase(text=item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item[:max_chars])
        if limit is not None and len(out) >= limit:
            break
    return out


def parse_description(*, text: str) -> tuple:
    """(description, discovered list) from a 'Description: ... Discovered: - ...' reply."""
    text = str(text or "")
    m = _DESC_RE.search(text)
    if m:
        description = " ".join(m.group(1).split())
    else:
        description = " ".join(re.split(r"discovered\s*:", text, flags=re.IGNORECASE)[0].split())
    d = _DISC_RE.search(text)
    return description, parse_bullets(block=d.group(1)) if d else []


def tokens_per_frame(*, width: int, height: int) -> int:
    return math.ceil(width / PATCH) * math.ceil(height / PATCH)


class TransitionDescriber:
    def __init__(self, *, vlm, max_new_tokens: int = 300, temperature: float = 0.0, max_chars: int = 800,
                 max_images: Optional[int] = None, max_model_len: Optional[int] = None,
                 parameters: dict[str, Any] = None) -> None:
        """max_images / max_model_len default to the vLLM server's limits in the config (vllm_max_images,
        vllm_max_model_len), the values the server is started with."""
        parameters = load_parameters(parameters)
        self.vlm, self.max_new_tokens, self.temperature, self.max_chars = vlm, max_new_tokens, temperature, max_chars
        self.max_images = int(max_images if max_images is not None else parameters["vllm_max_images"])
        self.max_model_len = int(max_model_len if max_model_len is not None else parameters["vllm_max_model_len"])
        self.n_calls = 0

    # ------------------------------------------------------------ windows

    def image_budget(self) -> int:
        return self.max_model_len - PROMPT_RESERVE - self.max_new_tokens

    def fit_frames(self, *, frames: list) -> tuple:
        """(frames, frames per window): frames are downscaled only if a single one is over the token budget."""
        w, h = max(f.width for f in frames), max(f.height for f in frames)
        budget = self.image_budget()
        per = tokens_per_frame(width=w, height=h)
        if per > budget:
            scale = math.sqrt(budget / per) * 0.95
            while tokens_per_frame(width=int(w * scale), height=int(h * scale)) > budget:
                scale *= 0.95
            frames = [f.resize((max(1, int(f.width * scale)), max(1, int(f.height * scale))), Image.BILINEAR)
                      for f in frames]
            per = tokens_per_frame(width=int(w * scale), height=int(h * scale))
        return frames, max(2, min(self.max_images, budget // per))

    @staticmethod
    def windows(*, n_frames: int, per_window: int) -> list:
        """(first, last) frame indices (0-based, inclusive) covering n_frames, consecutive windows sharing a frame."""
        if n_frames <= per_window:
            return [(0, n_frames - 1)]
        out, first = [], 0
        while first < n_frames - 1:
            last = min(n_frames - 1, first + per_window - 1)
            out.append((first, last))
            first = last
        return out

    # ------------------------------------------------------------ prompts

    @staticmethod
    def _action_lines(steps: list, first: int = 0) -> str:
        return "\n".join(f"{first + i + 1}. {action_line(action=s.action, valid=s.info.get('valid', True))}"
                         for i, s in enumerate(steps))

    def prompt(self, *, env_description: str, steps: list) -> str:
        return DESCRIBE_PROMPT.format(n_images=len(steps) + 1, env_description=env_description,
                                      n_actions=len(steps), actions=self._action_lines(steps))

    def __call__(self, *, env_description: str, start_frame, steps: list) -> Optional[dict]:
        """{"description", "discovered", "window_descriptions"} (None without steps). start_frame and each step's
        frame: EncodedImage or FrameRef (anything with .pil())."""
        if not steps:
            return None
        frames, per_window = self.fit_frames(frames=[start_frame.pil()] + [s.frame.pil() for s in steps])
        wins = self.windows(n_frames=len(frames), per_window=per_window)
        if len(wins) == 1:
            out = self.vlm.infer(texts=self.prompt(env_description=env_description, steps=steps), images=frames,
                                 max_new_tokens=self.max_new_tokens, temperature=self.temperature)
            self.n_calls += 1
            description, discovered = parse_description(text=out["output"])
            return {"description": description[:self.max_chars] or None, "discovered": discovered,
                    "window_descriptions": None}
        prompts, images = [], []
        for first, last in wins:
            prompts.append(WINDOW_PROMPT.format(
                n_images=last - first + 1, first=first + 1, last=last + 1, n_frames=len(frames),
                env_description=env_description, first_action=first + 1, last_action=last,
                actions=self._action_lines(steps[first:last], first=first)))
            images.append(frames[first:last + 1])
        outs = self.vlm.infer(texts=prompts, images=images, max_new_tokens=self.max_new_tokens,
                              temperature=self.temperature)["output"]
        window_descriptions = [{"frames": [first + 1, last + 1], "text": " ".join(str(o).split())[:self.max_chars]}
                               for (first, last), o in zip(wins, outs)]
        blocks = "\n".join(f"Frames {w['frames'][0]}-{w['frames'][1]}: {w['text']}" for w in window_descriptions)
        compose = COMPOSE_PROMPT.format(n_frames=len(frames), env_description=env_description,
                                        actions=self._action_lines(steps), windows=blocks)
        out = self.vlm.infer(texts=compose, images=None, max_new_tokens=self.max_new_tokens,
                             temperature=self.temperature)
        self.n_calls += len(wins) + 1
        description, discovered = parse_description(text=out["output"])
        return {"description": description[:self.max_chars] or None, "discovered": discovered,
                "window_descriptions": window_descriptions}
