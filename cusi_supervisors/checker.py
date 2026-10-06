"""Windowed trajectory summaries (GameBoyRL execution/supervisors/checker.py: window_trajectory,
summarise_trajectory_segments). The checker arm itself (AttemptCheckerSupervisor) is a
data-generation tool in GameBoyRL, not a benchmark arm, and is not ported.

Frames are each env step's frame_after (GameBoy: the screen; Android / Web: the unlabelled
screenshot); action lines are the steps' action labels.
"""
from typing import Any, Callable
from cusi_supervisors._format import step_name


def window_trajectory(env_steps: list, slice_prompt: str, *, game: str, call: Callable[..., Any],
                      max_new_tokens: int, domain, task: str = "", slice_size: int = 8) -> list:
    """Cut a trajectory into fixed-size windows and describe each, all in one batched call.
    Returns [((start, end), raw_output), ...]."""
    frames = [s.frame_after for s in env_steps]
    if not frames:
        return []

    total = len(env_steps)
    wants_actions = "[ACTION_SEQUENCE]" in slice_prompt
    action_lines_all = [f"  {i + 1}. {step_name(step, domain)}" for i, step in enumerate(env_steps)] if wants_actions else []

    segment_ranges, segment_prompts, segment_images = [], [], []
    for start in range(0, total, slice_size):
        end = min(start + slice_size, total)
        slice_actions = "\n".join(action_lines_all[start:end]) or "  (no actions taken)"
        prompt = (
            slice_prompt
            .replace("[GAME]", game)
            .replace("[TASK]", task)
            .replace("[START_IDX]", str(start + 1))
            .replace("[END_IDX]", str(end))
            .replace("[TOTAL]", str(total))
            .replace("[ACTION_SEQUENCE]", slice_actions)
        )
        segment_ranges.append((start, end))
        segment_prompts.append(prompt)
        segment_images.append(frames[start:end])

    outputs = call(texts=segment_prompts, images=segment_images, max_new_tokens=max_new_tokens)
    return list(zip(segment_ranges, outputs))


def summarise_trajectory_segments(env_steps: list, slice_prompt: str, game: str, task: str,
                                  call: Callable[..., Any], max_new_tokens: int, domain,
                                  max_frames_per_slice: int = 8) -> list:
    """One "Steps a-b: ..." summary per window (the "Segment summary:" line, else the reply)."""
    windows = window_trajectory(env_steps, slice_prompt, game=game, task=task, call=call,
                                max_new_tokens=max_new_tokens, domain=domain, slice_size=max_frames_per_slice)

    segment_summaries = []
    for (start, end), output in windows:
        for line in output.splitlines():
            if line.strip().lower().startswith("segment summary:"):
                summary = line.strip()[len("segment summary:"):].strip()
                segment_summaries.append(f"Steps {start + 1}-{end}: {summary}")
                break
        else:
            segment_summaries.append(f"Steps {start + 1}-{end}: {output.strip()}")
    return segment_summaries
