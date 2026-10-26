"""
Environment-agnostic prompts for the practice stages, with [DOMAIN]-style slots filled per environment.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    """How the shared prompts name an environment."""
    domain: str            # "a game of pokemon_red" / "an Android phone" / "a web browser"
    actor: str             # "player" / "user"
    screen: str            # "frame" / "screenshot"


def gameboy_domain(*, game: str) -> Domain:
    return Domain(domain=f"a game of {game}", actor="player", screen="frame")


ANDROID_DOMAIN = Domain(domain="an Android phone", actor="user", screen="screenshot")

WEB_DOMAIN = Domain(domain="a web browser", actor="user", screen="screenshot")


def fill(template: str, *, domain: Domain, **values) -> str:
    """Fill [DOMAIN]/[ACTOR]/[SCREEN] and any [KEY] given as KEY=value."""
    text = template.replace("[DOMAIN]", domain.domain).replace("[ACTOR]", domain.actor).replace("[SCREEN]", domain.screen)
    for key, value in values.items():
        text = text.replace(f"[{key}]", str(value))
    return text


# --------------------------------------------------------------------------- judge

DESCRIBE_SLICE_PROMPT = """You are watching [SCREEN]s [START_IDX]-[END_IDX] of [TOTAL] total [SCREEN]s from [DOMAIN].

Describe what the [ACTOR] does and what changes visually in this segment. Focus on actions taken and their outcomes. Do not assume any particular goal.

Respond in exactly this format:
Description: <concise description of the [ACTOR]'s actions and visual changes in this segment>
[STOP]"""

DESCRIBE_CONSOLIDATE_PROMPT = """You are consolidating segment descriptions from [DOMAIN] into a single complete trajectory description.

Segment descriptions (in chronological order), each labelled with the [SCREEN] range it covers:
[SEGMENT_DESCRIPTIONS]

Produce a single coherent description of the full trajectory from start to finish. Explicitly reference the [SCREEN] ranges (e.g. "[SCREEN]s 1-10", "[SCREEN]s 11-20") as you describe what happens, so the reader can tell which part of the trajectory each event belongs to. Keep these [SCREEN]-range references in the same form they appear in the segment labels above.

Respond in exactly this format:
Description: <complete description of the full trajectory, with [SCREEN] ranges referenced inline>
[STOP]"""

JUDGE_BINARY_PROMPT = """Task: "[TASK]"

A [ACTOR] attempted to complete this task on [DOMAIN]. Here is a description of what happened across the FULL trajectory:
"[DESCRIPTION]"
[ANSWER_NOTE]
The images show only the FINAL [SCREEN]s of the trajectory. Task completion may have occurred earlier and may not be visible in these images.

Did the [ACTOR] successfully complete the task at any point during the trajectory? Use the description as your primary evidence — if it mentions something that closely matches task completion, count it as success even if it is not visible in the final [SCREEN]s shown.
[GOAL_CONDITION_NOTE]
The description references [SCREEN] ranges (e.g. "[SCREEN]s 11-20"). Using these, identify the safe success point: the single [SCREEN] number by which the task has SURELY been achieved. Pick the earliest [SCREEN] you are confident the task is already complete. If the task was never completed, or you cannot tell from the description, respond with N/A.

Respond in exactly this format:
Reasoning: <your reasoning, referencing the description and any visual evidence>
Success: <yes or no>
Safe success point: <[SCREEN] number, or N/A if never completed or unknown>
[STOP]"""

JUDGE_GOAL_CONDITION_NOTE = """
The goal condition for this task is: "[GOAL_CONDITION]"

The goal condition is a strict guide, and only if the [ACTOR] has basically achieved the task with only minor, trivial differences from the goal condition should you consider it a success.
"""

# Fills [ANSWER_NOTE] when the agent ended with an answer.
JUDGE_ANSWER_NOTE = """
At the end, the [ACTOR] gave this final answer: "[ANSWER]"
If the task asks for information, judge whether this answer is correct and supported by what was seen.
"""

# --------------------------------------------------------------------------- critique

CRITIQUE_SLICE_PROMPT = """You are analysing a segment of a failed attempt to complete a task on [DOMAIN].

Task: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show [SCREEN]s [START_IDX]-[END_IDX] of the trajectory, from left to right.

Describe what happened in this segment: what the [ACTOR] did, what went wrong (if anything), and any observations relevant to why the task was not completed.

Respond in exactly this format:
Segment summary: <one or two sentences describing what happened in this segment>
[STOP]"""

CRITIQUE_CONSOLIDATE_PROMPT = """You are analysing a failed attempt to complete a task on [DOMAIN].

Task: "[TASK]"

Below are summaries of each segment of the failed trajectory:
[SEGMENT_SUMMARIES]

[PRIOR_HINT_BLOCK]Based on the full trajectory above, provide a concise hint for how to better approach the task on the next attempt.

Respond in exactly this format:
Critique: <what went wrong overall>
Hint: <one or two sentence hint for a better approach>
[STOP]"""

# --------------------------------------------------------------------------- dataset

AUGMENT_PARAPHRASE_PROMPT = """You are given a canonical task string:
"[CORE_TASK]"

Generate at least [N] diverse, valid paraphrases of this task. Vary the wording,
phrasing style, and structure but preserve the exact core meaning and level of specificity.
Use imperative tone throughout.

Respond in exactly this format (one paraphrase per line):
- <paraphrase 1>
- <paraphrase 2>
...
[STOP]"""
