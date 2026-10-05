"""Environment-agnostic prompts for the shared stages, ported from GameBoyRL.

Sources: vlm_scripts/propose_tasks_zeroshot.py (PROPOSE_PROMPT), execution/supervisors/prompts.py
(DESCRIBE_*, JUDGE_*, CRITIQUE_*), vlm_scripts/infer_guidance.py (*_GUIDANCE_PROMPT),
vlm_scripts/clean_practice.py (AUGMENT_PARAPHRASE_PROMPT, CLEAN_PROMPT*).

The wording is GameBoyRL's, with "a game of [GAME]" replaced by [DOMAIN] (filled from a
per-environment Domain), "player" by [ACTOR], and GameBoy-only examples moved into the Domain.
For GameBoy the Domain reproduces GameBoyRL's phrasing ("a game of pokemon_red", "player").
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    """How the shared prompts name an environment."""
    domain: str            # "a game of pokemon_red" / "an Android phone" / "a web browser"
    actor: str             # "player" / "agent"
    screen: str            # "frame" / "screenshot"
    step_examples: str     # imperative-step examples for the guidance prompt
    cue_example: str       # visual-cue example for the guidance prompt
    propose_rules: str = ""   # extra constraints for the propose prompt (one "- ..." line each)


def gameboy_domain(*, game: str) -> Domain:
    return Domain(
        domain=f"a game of {game}", actor="player", screen="frame",
        step_examples='"Press A to speak to the NPC", "Walk left towards the door"',
        cue_example=('instead of saying "walk upwards", say "when the character is past the tree line, walk upwards'
                     ' until the door is visible at the top of the frame."'))


_LABEL_RULE = ("- Described by what is visible (text, icons, position), never by the numeric labels in the"
               " element list: those numbers change from screen to screen\n")

ANDROID_DOMAIN = Domain(
    domain="an Android phone", actor="user", screen="screenshot", propose_rules=_LABEL_RULE,
    step_examples='"Open the Contacts app", "Tap the + button at the bottom right"',
    cue_example=('instead of saying "open the menu", say "when the contact list is showing, tap the three-dot icon'
                 ' at the top right until a menu with Settings appears."'))

WEB_DOMAIN = Domain(
    domain="a web browser", actor="user", screen="screenshot",
    propose_rules=_LABEL_RULE + ("- Possible without logging in, signing up, buying anything or entering personal"
                                 " information\n"),
    step_examples='"Type the query into the search box at the top", "Click the first result title"',
    cue_example=('instead of saying "search for it", say "when the home page is showing, type the query into the'
                 ' search box labelled Search at the top, until a page of results appears."'))


def fill(template: str, *, domain: Domain, **values) -> str:
    """Fill [DOMAIN]/[ACTOR]/[SCREEN] and any [KEY] given as KEY=value."""
    text = (template.replace("[DOMAIN]", domain.domain).replace("[ACTOR]", domain.actor)
            .replace("[SCREEN]", domain.screen).replace("[STEP_EXAMPLES]", domain.step_examples)
            .replace("[CUE_EXAMPLE]", domain.cue_example).replace("[PROPOSE_RULES]", domain.propose_rules))
    for key, value in values.items():
        text = text.replace(f"[{key}]", str(value))
    return text


# --------------------------------------------------------------------------- propose

PROPOSE_PROMPT = """You are observing the initial [SCREEN] of [DOMAIN].

You are an expert analyst of [DOMAIN]. Your job is to look carefully at this [SCREEN] and reason about:
- What is visible in the immediate surroundings (objects, characters, structures, interactable elements)
- What areas, items or features are accessible from this position
- What mechanisms or interactions are available given the action space: [ACTION_SPACE]
[TEXTS_BLOCK]
Propose an exhaustive list of distinct tasks that a [ACTOR] could meaningfully attempt to achieve starting from this exact state. Focus on tasks that are:
- Grounded in what is actually visible or reachable from this state
- Specific and concrete (not vague like "explore the area")
- Achievable as a single coherent goal
- Varied in scope (include both short and longer-horizon tasks)
[PROPOSE_RULES]
Respond in exactly this format:
Reasoning: <brief analysis of what is visible and what interactions are possible>
Tasks:
- <task 1>
- <task 2>
- <task 3>
...
[STOP]"""

# Fills [TEXTS_BLOCK] for environments with a text channel (the element list the native
# agent also sees). Empty for GameBoy [GB-OCR].
PROPOSE_TEXTS_BLOCK = """
The text elements on this [SCREEN] are:
[TEXTS]
"""

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

# Fills [ANSWER_NOTE] when the agent ended with an answer (WebVoyager ANSWER, M3A answer).
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

# --------------------------------------------------------------------------- guidance

SLICE_GUIDANCE_PROMPT = """You are an expert user of [DOMAIN]. You are given [SCREEN]s [START_IDX]-[END_IDX] (out of [TOTAL] total [SCREEN]s) from a trajectory, along with the task being accomplished:

Task: "[TASK]"

Describe what the [ACTOR] does in this section of the trajectory. Base your guidance strictly on what you can observe — do not invent details not visible.

Guidelines:
- Write each step as a short, clear imperative instruction (e.g. [STEP_EXAMPLES]).
- Order steps chronologically within this section.
- Be specific about directions, controls and targets when clearly visible.
- If the exact control labels are intuitive, describe the action instead.
- You must make sure that for each step, you explicitly refer to the visual cues in the [SCREEN] that indicate WHEN to do a particular step, and when to move on to the next. For example, [CUE_EXAMPLE] Be very specific about the visual cues that indicate what to do and when.

Respond in exactly this format:
Summary: <one sentence describing what happens in [SCREEN]s [START_IDX]-[END_IDX]>
Goal condition: <a visual description of the final [SCREEN] in this section>
Steps:
- <step 1>
- <step 2>
...
[STOP]"""

CONSOLIDATE_GUIDANCE_PROMPT = """You are consolidating partial guidance from multiple sections of a trajectory on [DOMAIN].

Task: "[TASK]"

Here are the descriptions of each section, in chronological order:
[SECTION_DESCRIPTIONS]

Combine these into a single, complete, coherent set of step-by-step guidance that a new [ACTOR] could follow from start to finish to accomplish the task.

Respond in exactly this format:
Summary: <one sentence describing the overall approach to complete the task>
Goal condition: <a visual description of the state that confirms task completion>
Steps:
- <step 1>
- <step 2>
...
[STOP]"""

# --------------------------------------------------------------------------- clean

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

CLEAN_PROMPT = """You are reviewing a single decision an agent made while operating [DOMAIN].

The agent was working toward this task:
"[TASK]"

It was shown the [SCREEN](s) provided as image(s) and given this prompt context:
[AGENT_PROMPT]

The agent's reasoning and action (its response) was:
[AGENT_RESPONSE]

Your job is to catch only CRITICAL errors. Consider two things:
- Is the agent's description of the [SCREEN] clearly wrong given the visual evidence?
- Is the chosen action clearly poor and unlikely to contribute toward completing the task?

Err strongly on the side of ACCEPT. Only REJECT if there is clear visual evidence that the
response badly misdescribes the [SCREEN], or that the action is likely counterproductive or
unlikely to advance the task. If you are unsure, ACCEPT.

Respond in exactly this format:
Reason: <one short sentence justifying your decision>
Decision: <ACCEPT or REJECT>
[STOP]"""

CLEAN_PROMPT_TRANSITION = """You are reviewing a single decision an agent made while operating [DOMAIN].

The agent was working toward this task:
"[TASK]"

It was shown the [SCREEN](s) provided as image(s) (every image EXCEPT the last) and given this prompt context:
[AGENT_PROMPT]

The agent's reasoning and action (its response) was:
[AGENT_RESPONSE]

The FINAL image is the [SCREEN] AFTER the agent's action was executed. Use the change from
the agent's [SCREEN](s) to this resulting [SCREEN] as your main evidence of whether the action helped.

Your job is to catch only CRITICAL errors. Consider:
- Is the agent's description of the [SCREEN] clearly wrong given the visual evidence?
- Does the action's visible effect (the before -> after change) clearly fail to advance the task, or move away from it?

Err strongly on the side of ACCEPT. Only REJECT if there is clear visual evidence that the
response badly misdescribes the [SCREEN], or that the resulting transition shows the action was
counterproductive or unlikely to advance the task. If you are unsure, ACCEPT.

Respond in exactly this format:
Reason: <one short sentence justifying your decision>
Decision: <ACCEPT or REJECT>
[STOP]"""

# M3A's summary rows are not actions; the reviewer judges the summary instead.
CLEAN_SUMMARY_PROMPT = """You are reviewing a step summary an agent wrote while operating [DOMAIN].

The agent was working toward this task:
"[TASK]"

It was shown the before and after [SCREEN]s provided as images and given this prompt context:
[AGENT_PROMPT]

The summary it wrote was:
[AGENT_RESPONSE]

Your job is to catch only CRITICAL errors: REJECT only if the summary clearly misdescribes what changed
between the two [SCREEN]s or what the action did. If you are unsure, ACCEPT.

Respond in exactly this format:
Reason: <one short sentence justifying your decision>
Decision: <ACCEPT or REJECT>
[STOP]"""
