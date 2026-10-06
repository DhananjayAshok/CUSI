"""Episode artifacts: one supervised episode (any arm, baseline included) as JSON + PNG + a step
video, readable by a debug panel without our classes (agents.md section 5). The content and
structure of GameBoyRL's archived SupervisorReport, minus the pickle.

    episodes/<task_id>/
      meta.json        task, env, models, executor arm, supervisor and its settings, outcome,
                       budget used, tokens and timings (supervisor / executor), answer, error,
                       supervisor state (plan, original_plan, step_log, ...) and env extras
      events.jsonl     the episode in order: {"kind": "supervisor", ...} and {"kind": "leg", ...}
      legs/<n>.jsonl   one line per executor call of leg n (prompt or chat, images, reply, tokens,
                       decision, and the steps it caused with their frames)
      frames/<k>.png   every image any call saw or any step produced, deduplicated
      episode.mp4      the step video

Image fields hold paths relative to the episode directory ("frames/12.png").

The step video: the leg's first frame, then the frame after every env step, in episode order across
legs, one frame per step (1 fps by default). Under each frame a caption bar: leg and target, step
number, the action, and INVALID / the env's error when the step failed. A decision that never
reached the env repeats the current screen with its INVALID caption, so gaps show. Frames are the
judge frames (Android / Web: the unlabelled screenshot); GameBoy's are upscaled 3x.
"""
import hashlib
import json
import os
import textwrap
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cusi_practice.records import EncodedImage, InvalidRecord, StepRecord, to_pil
from cusi_supervisors.report import LegEvent, SupervisorCall, SupervisorReport, jsonable

VIDEO_FPS = 1
GAMEBOY_SCALE = 3
#: StepRecord.extra keys worth keeping in the leg files (the rest are env internals).
STEP_EXTRA_KEYS = ("action_name", "frame_changed", "action_success", "low_level", "warning", "pdf", "url", "stale")


class FrameStore:
    """Writes each distinct image once, as frames/<k>.png; returns its relative path."""

    def __init__(self, *, directory: str) -> None:
        self.directory = directory
        os.makedirs(os.path.join(directory, "frames"), exist_ok=True)
        self._paths: dict = {}

    def __call__(self, image) -> Optional[str]:
        if image is None:
            return None
        encoded = EncodedImage.of(image)
        key = hashlib.sha1(encoded.png).hexdigest()
        if key not in self._paths:
            rel = f"frames/{len(self._paths)}.png"
            with open(os.path.join(self.directory, rel), "wb") as f:
                f.write(encoded.png)
            self._paths[key] = rel
        return self._paths[key]


def _step_json(step, frames: FrameStore) -> dict:
    if isinstance(step, InvalidRecord):
        return {"kind": "invalid", "reason": step.reason}
    return {"kind": "step", "action": step.action_text, "label": step.action_label(),
            "parsed_action": jsonable(step.parsed_action), "valid": step.valid, "error": step.error,
            "reward": step.reward, "frame_before": frames(step.frame_before), "frame_after": frames(step.frame_after),
            "extra": {k: jsonable(step.extra[k]) for k in STEP_EXTRA_KEYS if k in step.extra}}


def _call_json(call, frames: FrameStore) -> dict:
    out = {"tag": call.tag, "decision": call.decision, "images": [frames(i) for i in call.images],
           "response": call.response, "input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
           "seconds": getattr(call, "seconds", None), "steps": [_step_json(s, frames) for s in call.steps]}
    if call.messages is not None:
        out["messages"] = call.messages
    else:
        out["prompt"] = call.prompt
    return out


def write_episode(*, directory: str, meta: dict, report: SupervisorReport, gameboy: bool = False,
                  video: bool = True) -> None:
    """Write one episode's artifacts (see the module docstring)."""
    os.makedirs(os.path.join(directory, "legs"), exist_ok=True)
    frames = FrameStore(directory=directory)
    leg = 0
    with open(os.path.join(directory, "events.jsonl"), "w") as events:
        for i, event in enumerate(report.event_log):
            if isinstance(event, SupervisorCall):
                row = {"kind": "supervisor", "i": i, "stage": event.stage, "prompt": event.prompt,
                       "images": [frames(x) for x in event.images], "response": event.response,
                       "input_tokens": event.input_tokens, "output_tokens": event.output_tokens,
                       "seconds": event.seconds}
            else:
                leg += 1
                rep = event.report
                rel = f"legs/{leg}.jsonl"
                with open(os.path.join(directory, rel), "w") as f:
                    for call in rep.calls:
                        f.write(json.dumps(_call_json(call, frames), default=str) + "\n")
                row = {"kind": "leg", "i": i, "leg": leg, "task": rep.task, "target": event.target,
                       "hint": rep.hint, "final": event.final, "self_terminate": event.self_terminate,
                       "termination": rep.termination_reason, "n_steps": len(rep.steps),
                       "n_env_steps": len(rep.env_steps), "max_steps": rep.max_steps, "answer": rep.answer,
                       "final_reward": rep.final_reward, "error": rep.error, "seconds": event.seconds,
                       "initial_frame": frames(rep.initial_frame), "calls": rel}
            events.write(json.dumps(row, default=str) + "\n")
    full_meta = {**meta, "supervisor_name": report.supervisor_name, "supervisor_settings": report.init_kwargs,
                 **report.counts(), **report.timing(), "n_steps": report.n_steps, "n_invalid": report.n_invalid}
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(jsonable(full_meta), f, indent=1)
    if video:
        write_step_video(path=os.path.join(directory, "episode.mp4"), report=report,
                         scale=GAMEBOY_SCALE if gameboy else 1)


# --------------------------------------------------------------------------- the step video

def video_frames(*, report: SupervisorReport) -> list:
    """[(image, caption lines)] in episode order (see the module docstring)."""
    out = []
    legs = report.legs
    n_legs = len(legs)
    step_no = 0
    current = None
    for leg_no, event in enumerate(legs, start=1):
        rep = event.report
        where = f"leg {leg_no}/{n_legs}" + (f" | target: {event.target}" if event.target else
                                            (" | task" if n_legs > 1 else ""))
        if rep.initial_frame is not None:
            current = rep.initial_frame
            out.append((current, [where, f"start of leg ({rep.max_steps} steps)" + (
                f" | hint: {rep.hint}" if rep.hint else "")]))
        for call in rep.calls:
            if call.decision == "finish" and not call.steps:
                out.append((current, [where, f"FINISH: the agent declared this leg done"
                                             + (f" (answer: {rep.answer})" if rep.answer else "")]))
            for step in call.steps:
                step_no += 1
                if isinstance(step, InvalidRecord):
                    out.append((current, [where, f"step {step_no}: INVALID: {step.reason}"]))
                    continue
                current = step.frame_after
                line = f"step {step_no}: {_action_caption(step)}"
                if not step.valid:
                    line += f" | INVALID: {step.error or 'rejected by the env'}"
                out.append((current, [where, line]))
        out[-1][1].append(f"leg ended: {rep.termination_reason}")
    return out


def _action_caption(step: StepRecord) -> str:
    if "action_name" in step.extra:
        return str(step.extra["action_name"])
    if step.parsed_action:
        return json.dumps(step.parsed_action, default=str)
    text = step.action_text.strip()
    for line in text.splitlines():
        if line.strip().lower().startswith("action:"):
            return line.strip()
    return text.splitlines()[-1] if text else "(none)"


def _font(size: int):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), size)
    except Exception:
        return ImageFont.load_default()


def captioned(*, image, lines: list, scale: int = 1) -> Image.Image:
    """The frame (upscaled by scale) above a black caption bar holding the wrapped lines."""
    pil = to_pil(image)
    if scale != 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    size = max(12, pil.width // 45)
    font = _font(size)
    chars = max(20, int(pil.width / (size * 0.62)))
    wrapped = [w for line in lines for w in (textwrap.wrap(line, chars) or [""])][:8]
    bar = int(len(wrapped) * size * 1.3) + size
    canvas = Image.new("RGB", (pil.width, pil.height + bar), (0, 0, 0))
    canvas.paste(pil, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for k, line in enumerate(wrapped):
        colour = (255, 120, 120) if "INVALID" in line or "error" in line.lower() else (255, 255, 255)
        draw.text((size // 2, pil.height + size // 2 + int(k * size * 1.3)), line, fill=colour, font=font)
    return canvas


def write_step_video(*, path: str, report: SupervisorReport, scale: int = 1, fps: int = VIDEO_FPS) -> None:
    import imageio.v2 as imageio
    items = [(img, lines) for img, lines in video_frames(report=report) if img is not None]
    if not items:
        return
    pictures = [captioned(image=img, lines=lines, scale=scale) for img, lines in items]
    width = max(p.width for p in pictures)
    height = max(p.height for p in pictures)
    width, height = width + width % 2, height + height % 2
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_log_level="error") as writer:
        for p in pictures:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:p.height, :p.width] = np.asarray(p)
            writer.append_data(frame)
