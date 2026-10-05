"""Generic records of what an executor did, shared by all three environments.

Mirrors GameBoyRL's execution/report.py (EnvironmentStepRecord, InvalidStepRecord,
ExecutorVLMCallRecord, ExecutorReport), with environment-specific fields replaced by
generic ones, so a pickle loads without any environment's classes:

    StepRecord   one env.step: frames before/after, the action text, parsed_action, valid,
                 error, reward, plus a small `extra` dict (e.g. GameBoy's frame_changed).
    CallRecord   one model call: tag, images, prompt (single-turn) or messages (chat),
                 response, token counts, and the steps that call caused.
    LegReport    one executor run on one task: every call (in order), termination reason,
                 the agent's answer (if any).

Images are stored PNG-encoded (EncodedImage): Android screenshots are 7.7 MB raw, and an
episode holds hundreds of them.
"""
import io
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import numpy as np
from PIL import Image

#: Tags of the calls that choose an action (training rows come from these).
ACTION_TAGS = {"action"}
#: M3A's step summary call: kept as a training row too (M3A writes its summaries at test time).
SUMMARY_TAG = "summary"
#: Tags whose calls become dataset rows.
DATASET_TAGS = ACTION_TAGS | {SUMMARY_TAG}


class EncodedImage:
    """An RGB image held as PNG bytes. Build with EncodedImage.of(array_or_pil)."""

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
    """PIL RGB from an HxW, HxWx1, HxWx3 array, a PIL image or an EncodedImage.

    Colour is kept: GameBoyRL's converter (utils/vlm.py) took channel 0 only, which would
    grey out Android/web screenshots."""
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


@dataclass
class StepRecord:
    """One env.step. frame_before/frame_after are what the judge sees (the raw screenshot
    where an environment has one, else the observation frame)."""
    frame_before: EncodedImage
    frame_after: EncodedImage
    action_text: str
    parsed_action: Optional[dict]
    valid: bool
    error: Optional[str]
    reward: float = 0.0
    extra: dict = field(default_factory=dict)

    def action_label(self) -> str:
        """Short human-readable action name, for prompts that list actions taken."""
        if "action_name" in self.extra:
            return str(self.extra["action_name"])
        if self.parsed_action:
            return ", ".join(f"{k}={v}" for k, v in self.parsed_action.items())
        return self.action_text.strip().splitlines()[-1] if self.action_text.strip() else "(none)"


@dataclass
class InvalidRecord:
    """A deciding call that did not reach the environment (unparseable, or rejected by the
    executor before env.step), as GameBoyRL's InvalidStepRecord."""
    response: str
    reason: str


@dataclass
class CallRecord:
    """One model call.

    Single-turn calls set `prompt` (text) and `images` (in order). Chat calls (WebVoyager)
    set `messages`: role/content dicts whose content is a str or a list of
    {"type": "text", "text": ...} / {"type": "image", "image": <index into images>} parts."""
    tag: str
    images: list
    response: str
    prompt: Optional[str] = None
    messages: Optional[list] = None
    steps: list = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    #: For deciding calls: "step" (sent to the env), "finish" (the agent declared the task
    #: done), "invalid" (never reached the env). None for auxiliary calls.
    decision: Optional[str] = None

    @property
    def env_steps(self) -> list:
        return [s for s in self.steps if isinstance(s, StepRecord)]

    def accepted_by_env(self) -> bool:
        """The legality filter: a finish, or a step the environment accepted."""
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
    #: The env's reward on the last env step (test mode: the benchmark's success signal).
    final_reward: Optional[float] = None
    #: Set when the leg ended on a model error (Executor.run(stop_on_model_error=True)).
    error: Optional[str] = None

    @property
    def steps(self) -> list:
        return [s for c in self.calls for s in c.steps]

    @property
    def env_steps(self) -> list:
        return [s for s in self.steps if isinstance(s, StepRecord)]

    def frames(self) -> list:
        """initial frame + the frame after each env step."""
        out = [self.initial_frame] if self.initial_frame is not None else []
        return out + [s.frame_after for s in self.env_steps]
