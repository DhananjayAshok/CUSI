"""
Environment-agnostic records of what an executor did: steps, model calls and legs.
"""
import io
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import numpy as np
from PIL import Image

ACTION_TAGS = {"action"}
#: M3A writes its step summaries at test time, so they are training rows too.
SUMMARY_TAG = "summary"
DATASET_TAGS = ACTION_TAGS | {SUMMARY_TAG}


class EncodedImage:
    """An RGB image held as PNG bytes (raw screenshots are too big to keep per step)."""

    __slots__ = ("png", "size")

    def __init__(self, *, png: bytes, size: tuple) -> None:
        self.png = png
        self.size = size   # (width, height)

    @classmethod
    def of(cls, image: Union[np.ndarray, Image.Image, "EncodedImage"]) -> "EncodedImage":
        if isinstance(image, EncodedImage):
            return image
        pil = to_pil(image)
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=False, compress_level=3)
        return cls(png=buf.getvalue(), size=pil.size)

    def pil(self) -> Image.Image:
        return Image.open(io.BytesIO(self.png)).convert("RGB")

    def numpy(self) -> np.ndarray:
        return np.asarray(self.pil(), dtype=np.uint8)

    def __getstate__(self):
        return {"png": self.png, "size": self.size}

    def __setstate__(self, state):
        self.png = state["png"]
        self.size = state["size"]


def to_pil(image: Any) -> Image.Image:
    """PIL RGB from an array, a PIL image or an EncodedImage, keeping colour."""
    if isinstance(image, EncodedImage):
        return image.pil()
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(np.ascontiguousarray(arr))


def per_prompt_token_counts(*, meta: dict, n_prompts: int) -> list:
    """A batched call's (input, output) token counts per prompt; an unsplit total goes on the first."""
    def spread(value):
        if isinstance(value, list) and len(value) == n_prompts:
            return list(value)
        if isinstance(value, list):
            known = [v for v in value if v is not None]
            value = sum(known) if known else None
        return [value] + [None if value is None else 0] * (n_prompts - 1)
    return list(zip(spread(meta.get("input_tokens")), spread(meta.get("output_tokens"))))


@dataclass
class StepRecord:
    """One env.step; the frames are what the judge sees."""
    frame_before: EncodedImage
    frame_after: EncodedImage
    action_text: str
    parsed_action: Optional[dict]
    valid: bool
    error: Optional[str]
    reward: float = 0.0
    extra: dict = field(default_factory=dict)

    def action_label(self) -> str:
        if "action_name" in self.extra:
            return str(self.extra["action_name"])
        if self.parsed_action:
            return ", ".join(f"{k}={v}" for k, v in self.parsed_action.items())
        return self.action_text.strip().splitlines()[-1] if self.action_text.strip() else "(none)"


@dataclass
class InvalidRecord:
    """A deciding call that never reached the environment."""
    response: str
    reason: str


@dataclass
class CallRecord:
    """One model call: single-turn sets `prompt`; chat sets `messages`, whose image parts index `images`."""
    tag: str
    images: list
    response: str
    prompt: Optional[str] = None
    messages: Optional[list] = None
    steps: list = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    #: "step", "finish" or "invalid" for deciding calls; None for auxiliary calls.
    decision: Optional[str] = None
    #: A batched call's total goes on its first record.
    seconds: Optional[float] = None

    @property
    def env_steps(self) -> list:
        return [s for s in self.steps if isinstance(s, StepRecord)]

    def accepted_by_env(self) -> bool:
        """A finish, or a step the environment accepted."""
        if self.decision == "finish":
            return True
        return self.decision == "step" and any(s.valid for s in self.env_steps)


@dataclass
class LegReport:
    """One executor run (a "leg") on one task."""
    env_name: str
    task: str
    hint: Optional[str]
    max_steps: int
    calls: list = field(default_factory=list)
    termination_reason: Optional[str] = None
    answer: Optional[str] = None
    initial_frame: Optional[EncodedImage] = None
    #: In test mode, the benchmark's success signal.
    final_reward: Optional[float] = None
    error: Optional[str] = None

    @property
    def steps(self) -> list:
        return [s for c in self.calls for s in c.steps]

    @property
    def env_steps(self) -> list:
        return [s for s in self.steps if isinstance(s, StepRecord)]

    def frames(self) -> list:
        """The initial frame, then the frame after each env step."""
        out = [self.initial_frame] if self.initial_frame is not None else []
        return out + [s.frame_after for s in self.env_steps]
