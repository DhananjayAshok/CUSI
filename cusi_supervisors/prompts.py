"""Every prompt the supervisors send, per env.

GameBoy: GameBoyRL's own texts (execution/supervisors/prompts.py), copied verbatim — only the
prompts the supervisor arms send (plan, judge, regression, hint, plan-flaw, and the knowledge
selection of info_subgoal); the checker arm's CRITIQUE_* / DESCRIBE_* / JUDGE_BINARY_* are not
ported. tests/supervisor_prompts_test.py asserts they equal GameBoyRL's.

Android / Web: GENERIC_* below, the same prompts with games, players and buttons reworded for an
agent on a phone or in a browser (agents.md decision 1), filled per env by prompts_for().
"""
from dataclasses import dataclass

# ===========================================================================================
# GameBoy: GameBoyRL's texts, verbatim
# ===========================================================================================

PLAN_PROMPT = """You are planning how a player should complete a task in a game of [GAME].

Task: "[TASK]"

The image is the screen the player is looking at right now.

Here is what has been learned from past playthroughs of this game that may be relevant:
[INSIGHTS]

Break the task into an ordered sequence of steps, separated by the token [STEP].

Each step will be given to a player who CANNOT see the other steps and does not know how many remain. They see only the current screen and the step you wrote. So each step must stand entirely on its own.

Requirements for every step:

- Describe the step by what is VISIBLE on screen: objects, icons, cursors, doors, characters, menu entries, text. Refer to things the player can point at.
- Do NOT name buttons or directions. Write "move the cursor to the coat" rather than "press RIGHT twice to reach the coat", and "select the hand tool" rather than "press A". Which button achieves it is the player's problem, and the button that worked in a past playthrough may be wrong from this screen.
- Every step MUST carry a termination condition that is visually checkable — a state of the screen the player can look at and confirm. Write it as "... until <what the screen shows>". If you cannot name a visible condition that ends the step, the step is too vague: merge it into a neighbour or rewrite it.
- One step should be one coherent sub-goal, not a single input and not the whole task.
- Prefer few steps. Three or four good steps beat ten brittle ones.

Do not include a step for something the screen shows is already done.

Respond in exactly this format:
Plan: <step one, ending in a visible condition> [STEP] <step two, ending in a visible condition> [STEP] <...>
[STOP]"""

FILTER_INSIGHTS_PROMPT = """You are pruning recorded knowledge about [GAME] down to what could matter for one task.

The player's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is everything recorded from past playthroughs that was retrieved for this situation:
[CANDIDATES]

Some of it will be about things that have nothing to do with this task or this place — advice about talking to a character when nobody is here, about a menu that this task never opens, about a room the player is not in and will not enter. That is what you are removing.

Keep an item if it could plausibly matter at ANY point while doing this task, not only on the screen as it looks this instant. The player will move, open menus and change rooms while working, and knowledge about where they are heading is exactly what is worth keeping. Something you drop is gone for the whole task.

So: drop only what is clearly about something absent and unrelated. **If you are unsure, KEEP it.** Removing one useful item costs more than leaving three useless ones.

Respond in exactly this format:
Keep: <comma-separated numbers, or ALL>
[STOP]"""

DISTILL_INSIGHTS_PROMPT = """You are consolidating what is known about [GAME] into a briefing for one task.

The player's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is the knowledge kept from past playthroughs. It was recorded piecemeal, by different runs, so it repeats itself, contradicts itself in places, and states the same thing at several levels of detail:
[CANDIDATES]

Rewrite it as a short, ordered list of concrete statements.

- **Aggregate.** Where several items describe one thing, merge them into a single statement that carries every specific detail any of them had. Prefer the most specific version: if one says "an icon in the toolbar" and another says "the third icon from the left", the merged statement says the third icon from the left.
- **Cut redundancy.** Two items that say the same thing become one. An item that is a vaguer restatement of another is dropped entirely.
- **Be concrete.** Name the object, the place on the screen, the observable result. Drop anything that survives only as generic advice — "be careful", "explore thoroughly", "pay attention to the surroundings" — that is not knowledge, it is filler.
- **Stay faithful.** Every statement must be supported by the items above. Do not add knowledge, do not resolve a contradiction by inventing a third version, and do not promote a guess into a fact. If two items genuinely disagree, say so in one statement and keep both readings.
- Do not narrow a statement to only what is on this screen. The player will move and change rooms while doing this task, and knowledge about where they are going still belongs here.

Respond in exactly this format, one statement per line:
Insights:
- <statement>
- <statement>
[STOP]"""

JUDGE_SLICE_PROMPT = """You are examining a segment of a player's attempt at one step of a plan in a game of [GAME].

The step they were asked to complete: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show frames [START_IDX]-[END_IDX] of the attempt, from left to right.

Describe what visibly changed on screen across this segment, and whether anything in it shows the step's termination condition being met. Report what you can see, not what you assume the player intended.

Respond in exactly this format:
Segment summary: <one or two sentences describing what visibly happened>
[STOP]"""

JUDGE_CONSOLIDATE_PROMPT = """You are deciding whether a player completed one step of a plan in a game of [GAME].

The step they were asked to complete: "[TASK]"

Summaries of each segment of their attempt:
[SEGMENT_SUMMARIES]

The image is the screen as it stands NOW, at the end of the attempt. It is your primary evidence: the step is complete if and only if this screen shows its termination condition met.

The player stopped because: [STOP_REASON]. Note that a player who declared itself finished may be wrong — judge the screen, not the claim.

Answering "yes" when the step is not done sends the plan onward from a state it does not expect, and everything after it is built on a false premise. Answering "no" when it is done wastes the step budget repeating work. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to what is visible in the final screen>
Complete: <yes or no>
[STOP]"""

REGRESSION_CHECK_PROMPT = """You are checking whether a player of [GAME] has undone progress they had already made.

Earlier in this task they completed this step:
"[PREVIOUS_STEP]"

[EARLIER_STEPS_BLOCK]You are given two images. The FIRST is the screen at the moment that step was judged complete. The SECOND is the screen now, after a later step was attempted and failed.

What happened in between:
[SEGMENT_SUMMARIES]

Compare the two screens. Has the state that made the earlier step complete been lost? Examples of losing it: a tool that was selected is no longer selected, a menu that was open has closed, a door that was opened is shut again, an item that was held has been put back, the player has left the room they had reached.

Judge only what the two images show. Do not guess from the actions described — if the second screen still shows the earlier step's result, it was not undone, however erratic the play looks. If the images are too similar to tell, say no.

Respond in exactly this format:
Reasoning: <one or two sentences comparing the two screens>
Undone: <yes or no>
What was lost: <if yes, name the specific thing that is no longer true; otherwise write NONE>
[STOP]"""

RESUME_HINT_PROMPT = """You are advising a player of [GAME] who has just failed to complete one step of a plan and is about to try again.

The step: "[TASK]"

Summaries of what they just did:
[SEGMENT_SUMMARIES]

Why it is judged incomplete: [JUDGEMENT]

[REGRESSION_BLOCK]What past playthroughs of this game recorded that may bear on this:
[INSIGHTS]

[PRIOR_HINT_BLOCK]The image is the screen the player is looking at RIGHT NOW. They are NOT starting over — the game is exactly as this screen shows, including any progress or damage from the failed attempt.

Work in two parts.

**First, diagnose.** Say what is actually going wrong, using the reasons the player gave for each button beside what the frames show happened. Name the mechanism, not the symptom: not "they failed to select the tool" but why the presses that should have selected it did not.

**Then instruct.** Unlike the plan, which describes goals without mentioning controls, your hint names the actual controls: UP, DOWN, LEFT, RIGHT, A, B, START. Say **what each button does towards this goal** — which one moves the cursor, which one confirms, which one backs out of the menu they are stuck in. Use the recorded knowledge above wherever it names a control or what it does; that is what it is for.

**Do not give a count or a sequence.** Not "press DOWN four times, then A". The player acts one button at a time and looks at the screen again after each one, so a recipe written from this screen is wrong by its second step, and a player following it stops watching the screen. Give them the function of each control and the visible condition that tells them to stop: "DOWN moves the selection down the list — keep going until KEY1 is the circled entry, then A confirms it."

Requirements:

- The instruction must start from THIS screen. If the failed attempt left the player somewhere unexpected, say which button gets them out of it first.
- Correct a false belief explicitly before instructing: "you are two tiles left of the icon, not on it — RIGHT moves the cursor towards it, and A selects once it is highlighted" beats restating the goal.
- Do not repeat an instruction the summaries show already failed. If pressing A did nothing three times, do not say press A; say which button does the thing they were trying to do.
- Tie every button to an effect the player can see. A button named without saying what it changes on screen is no more useful than the plan step was.

Respond in exactly this format:
Diagnosis: <one or two sentences naming what is actually going wrong>
Hint: <which buttons do what towards this goal, and the visible condition to stop at; two sentences at most, no counts>
[STOP]"""

PLAN_FLAW_PROMPT = """You are reviewing whether a plan for a task in [GAME] is still worth following.

The overall task: "[OVERALL_TASK]"

The plan, with progress marked:
[PLAN_BLOCK]

The current step has just failed. What happened:
[FAILURE_HISTORY]

Why it is judged incomplete: [JUDGEMENT]
[REGRESSION_LINE]
What past playthroughs of this game recorded:
[INSIGHTS]

The image is the screen the player is looking at right now.

A plan can fail for two very different reasons, and you are deciding which:

**The plan is sound, the player is fumbling it.** The steps describe the right route; the player misread the screen, pressed the wrong control, or acted on the wrong object. A better hint fixes this. Answer **no**.

**The plan is wrong.** The screens show something the plan did not anticipate: the route it assumes does not exist, an object it names is not there, a step depends on a state that cannot be reached from here, the game works differently from what the plan assumed, or the player is somewhere the plan has no path from. No hint fixes this, because the player is being asked to do the wrong thing. Answer **yes**.

Be strict. Repeated failure alone is not evidence of a bad plan — a fumbled step fails repeatedly too. You need something visible on the screens that the plan is incompatible with. If you cannot name that thing, answer no.

If you answer yes, write a replacement for the current step and everything after it. Steps already marked DONE are finished and must not be re-planned; start from where the player is now. Keep the original plan's rules: describe what is VISIBLE, never name buttons or directions, and end every step with a visually checkable condition ("... until <what the screen shows>").

Respond in exactly this format:
Reasoning: <what on the screens does or does not contradict the plan>
Flawed: <yes or no>
Plan: <the replacement steps separated by [STEP], or NONE if not flawed>
[STOP]"""

RELEVANCE_PROMPT = """You are deciding whether a piece of recorded knowledge about [GAME] is relevant to the situation a player is in right now.

The player's current task is: "[TASK]"

Here is the recorded entry:
[ENTRY]

[FRAME_NOTE]

Could this entry's knowledge be relevant to the player's current task on this current screen? Answer yes only if the entry genuinely fits the situation — the same or a very similar [KIND]. [EVIDENCE_NOTE]

Answering yes to something that does not fit produces a misleading hint, which is worse than no hint at all. Answering no to something that does fit wastes knowledge that was already paid for. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to the frames>
Relevant: <yes or no>
[STOP]"""

# ===========================================================================================
# Android / Web: the same prompts reworded for an agent using a phone or a browser.
# [WHERE] ("on an Android phone" / "in a web browser"), [SUBJECT] (what knowledge is about),
# [CONTROLS], [PLAN_EXAMPLE], [REGRESSION_EXAMPLES], [HINT_EXAMPLE] and [STUCK_EXAMPLE] are filled
# from the env's SupervisorDomain by prompts_for(); every other placeholder is the supervisor's.
# Structure, rules and response formats are GameBoyRL's; what changes is the wording about
# games, players and buttons, and the planner rule (element numbers instead of buttons).
# ===========================================================================================

GENERIC_PLAN_PROMPT = """You are planning how an agent should complete a task [WHERE].

Task: "[TASK]"

The image is the screen the agent is looking at right now.

Here is what has been learned from past attempts that may be relevant:
[INSIGHTS]

Break the task into an ordered sequence of steps, separated by the token [STEP].

Each step will be given to an agent who CANNOT see the other steps and does not know how many remain. They see only the current screen and the step you wrote. So each step must stand entirely on its own.

Requirements for every step:

- Describe the step by what is VISIBLE on screen: apps, pages, buttons, fields, links, icons, menus, text. Refer to things the agent can point at.
- Do NOT name element numbers or exact actions. [PLAN_EXAMPLE] The numbered labels change on every screen, and an action that worked once may be wrong from another screen; describe the element by its visible text, icon or position.
- Every step MUST carry a termination condition that is visually checkable — a state of the screen the agent can look at and confirm. Write it as "... until <what the screen shows>". If you cannot name a visible condition that ends the step, the step is too vague: merge it into a neighbour or rewrite it.
- One step should be one coherent sub-goal, not a single action and not the whole task.
- Prefer few steps. Three or four good steps beat ten brittle ones.

Do not include a step for something the screen shows is already done.

Respond in exactly this format:
Plan: <step one, ending in a visible condition> [STEP] <step two, ending in a visible condition> [STEP] <...>
[STOP]"""

GENERIC_FILTER_INSIGHTS_PROMPT = """You are pruning recorded knowledge about [SUBJECT] down to what could matter for one task.

The agent's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is everything recorded from past attempts that was retrieved for this situation:
[CANDIDATES]

Some of it will be about things that have nothing to do with this task or this place — advice about a dialog that is not open, about a menu that this task never opens, about an app or page the agent is not on and will not visit. That is what you are removing.

Keep an item if it could plausibly matter at ANY point while doing this task, not only on the screen as it looks this instant. The agent will open menus, change screens and move between pages while working, and knowledge about where they are heading is exactly what is worth keeping. Something you drop is gone for the whole task.

So: drop only what is clearly about something absent and unrelated. **If you are unsure, KEEP it.** Removing one useful item costs more than leaving three useless ones.

Respond in exactly this format:
Keep: <comma-separated numbers, or ALL>
[STOP]"""

GENERIC_DISTILL_INSIGHTS_PROMPT = """You are consolidating what is known about [SUBJECT] into a briefing for one task.

The agent's task is: "[TASK]"

The image is the screen they are looking at right now.

Here is the knowledge kept from past attempts. It was recorded piecemeal, by different runs, so it repeats itself, contradicts itself in places, and states the same thing at several levels of detail:
[CANDIDATES]

Rewrite it as a short, ordered list of concrete statements.

- **Aggregate.** Where several items describe one thing, merge them into a single statement that carries every specific detail any of them had. Prefer the most specific version: if one says "a button in the toolbar" and another says "the third icon from the left in the toolbar", the merged statement says the third icon from the left.
- **Cut redundancy.** Two items that say the same thing become one. An item that is a vaguer restatement of another is dropped entirely.
- **Be concrete.** Name the element, the place on the screen, the observable result. Drop anything that survives only as generic advice — "be careful", "explore thoroughly", "pay attention to the page" — that is not knowledge, it is filler.
- **Stay faithful.** Every statement must be supported by the items above. Do not add knowledge, do not resolve a contradiction by inventing a third version, and do not promote a guess into a fact. If two items genuinely disagree, say so in one statement and keep both readings.
- Do not narrow a statement to only what is on this screen. The agent will move between screens while doing this task, and knowledge about where they are going still belongs here.

Respond in exactly this format, one statement per line:
Insights:
- <statement>
- <statement>
[STOP]"""

GENERIC_JUDGE_SLICE_PROMPT = """You are examining a segment of an agent's attempt at one step of a plan [WHERE].

The step they were asked to complete: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show the screen after each of steps [START_IDX]-[END_IDX] of the attempt, in order.

Describe what visibly changed on screen across this segment, and whether anything in it shows the step's termination condition being met. Report what you can see, not what you assume the agent intended.

Respond in exactly this format:
Segment summary: <one or two sentences describing what visibly happened>
[STOP]"""

GENERIC_JUDGE_CONSOLIDATE_PROMPT = """You are deciding whether an agent completed one step of a plan [WHERE].

The step they were asked to complete: "[TASK]"

Summaries of each segment of their attempt:
[SEGMENT_SUMMARIES]

The image is the screen as it stands NOW, at the end of the attempt. It is your primary evidence: the step is complete if and only if this screen shows its termination condition met.

The agent stopped because: [STOP_REASON]. Note that an agent that declared itself finished may be wrong — judge the screen, not the claim.

Answering "yes" when the step is not done sends the plan onward from a state it does not expect, and everything after it is built on a false premise. Answering "no" when it is done wastes the step budget repeating work. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to what is visible in the final screen>
Complete: <yes or no>
[STOP]"""

GENERIC_REGRESSION_CHECK_PROMPT = """You are checking whether an agent working [WHERE] has undone progress they had already made.

Earlier in this task they completed this step:
"[PREVIOUS_STEP]"

[EARLIER_STEPS_BLOCK]You are given two images. The FIRST is the screen at the moment that step was judged complete. The SECOND is the screen now, after a later step was attempted and failed.

What happened in between:
[SEGMENT_SUMMARIES]

Compare the two screens. Has the state that made the earlier step complete been lost? Examples of losing it: [REGRESSION_EXAMPLES]

Judge only what the two images show. Do not guess from the actions described — if the second screen still shows the earlier step's result, it was not undone, however erratic the actions look. If the images are too similar to tell, say no.

Respond in exactly this format:
Reasoning: <one or two sentences comparing the two screens>
Undone: <yes or no>
What was lost: <if yes, name the specific thing that is no longer true; otherwise write NONE>
[STOP]"""

GENERIC_RESUME_HINT_PROMPT = """You are advising an agent working [WHERE] who has just failed to complete one step of a plan and is about to try again.

The step: "[TASK]"

Summaries of what they just did:
[SEGMENT_SUMMARIES]

Why it is judged incomplete: [JUDGEMENT]

[REGRESSION_BLOCK]What past attempts recorded that may bear on this:
[INSIGHTS]

[PRIOR_HINT_BLOCK]The image is the screen the agent is looking at RIGHT NOW. They are NOT starting over — the screen is exactly as shown, including any progress or damage from the failed attempt.

Work in two parts.

**First, diagnose.** Say what is actually going wrong, using the reasons the agent gave for each action beside what the screens show happened. Name the mechanism, not the symptom: not "they failed to open the menu" but why the actions that should have opened it did not.

**Then instruct.** Unlike the plan, which describes goals without naming actions, your hint names the actual actions: [CONTROLS]. Say **which action on which visible element moves towards this goal** — what to tap or click, what to type and into which field, when to scroll, when to go back. Use the recorded knowledge above wherever it names an action or an element; that is what it is for.

**Do not give a count, a sequence or an element number.** Not "[STUCK_EXAMPLE]". The agent acts one action at a time and looks at the screen again after each one, and the numbered labels change on every screen, so a recipe written from this screen is wrong by its second step. Give them the element by its visible text, icon or position, the action that works on it, and the visible condition that tells them to stop: "[HINT_EXAMPLE]"

Requirements:

- The instruction must start from THIS screen. If the failed attempt left the agent somewhere unexpected, say which action gets them out of it first.
- Correct a false belief explicitly before instructing: "the search box is the field at the top of the page, not the banner below it — click the field, then type the query" beats restating the goal.
- Do not repeat an instruction the summaries show already failed. If clicking an element did nothing three times, do not say click it; say which other element or action does the thing they were trying to do.
- Tie every action to an effect the agent can see. An action named without saying what it changes on screen is no more useful than the plan step was.

Respond in exactly this format:
Diagnosis: <one or two sentences naming what is actually going wrong>
Hint: <which action on which visible element, towards this goal, and the visible condition to stop at; two sentences at most, no counts, no element numbers>
[STOP]"""

GENERIC_PLAN_FLAW_PROMPT = """You are reviewing whether a plan for a task [WHERE] is still worth following.

The overall task: "[OVERALL_TASK]"

The plan, with progress marked:
[PLAN_BLOCK]

The current step has just failed. What happened:
[FAILURE_HISTORY]

Why it is judged incomplete: [JUDGEMENT]
[REGRESSION_LINE]
What past attempts recorded:
[INSIGHTS]

The image is the screen the agent is looking at right now.

A plan can fail for two very different reasons, and you are deciding which:

**The plan is sound, the agent is fumbling it.** The steps describe the right route; the agent misread the screen, chose the wrong action, or acted on the wrong element. A better hint fixes this. Answer **no**.

**The plan is wrong.** The screens show something the plan did not anticipate: the route it assumes does not exist, an element it names is not there, a step depends on a state that cannot be reached from here, the app or site works differently from what the plan assumed, or the agent is somewhere the plan has no path from. No hint fixes this, because the agent is being asked to do the wrong thing. Answer **yes**.

Be strict. Repeated failure alone is not evidence of a bad plan — a fumbled step fails repeatedly too. You need something visible on the screens that the plan is incompatible with. If you cannot name that thing, answer no.

If you answer yes, write a replacement for the current step and everything after it. Steps already marked DONE are finished and must not be re-planned; start from where the agent is now. Keep the original plan's rules: describe what is VISIBLE, never name element numbers or exact actions, and end every step with a visually checkable condition ("... until <what the screen shows>").

Respond in exactly this format:
Reasoning: <what on the screens does or does not contradict the plan>
Flawed: <yes or no>
Plan: <the replacement steps separated by [STEP], or NONE if not flawed>
[STOP]"""

GENERIC_RELEVANCE_PROMPT = """You are deciding whether a piece of recorded knowledge about [SUBJECT] is relevant to the situation an agent is in right now.

The agent's current task is: "[TASK]"

Here is the recorded entry:
[ENTRY]

[FRAME_NOTE]

Could this entry's knowledge be relevant to the agent's current task on this current screen? Answer yes only if the entry genuinely fits the situation — the same or a very similar [KIND]. [EVIDENCE_NOTE]

Answering yes to something that does not fit produces a misleading hint, which is worse than no hint at all. Answering no to something that does fit wastes knowledge that was already paid for. Judge honestly in both directions.

Respond in exactly this format:
Reasoning: <one or two sentences, referring to the frames>
Relevant: <yes or no>
[STOP]"""


# ===========================================================================================
# Per-env wording, and the prompt set for an env
# ===========================================================================================

PROMPT_NAMES = ("PLAN_PROMPT", "FILTER_INSIGHTS_PROMPT", "DISTILL_INSIGHTS_PROMPT", "JUDGE_SLICE_PROMPT",
                "JUDGE_CONSOLIDATE_PROMPT", "REGRESSION_CHECK_PROMPT", "RESUME_HINT_PROMPT", "PLAN_FLAW_PROMPT",
                "RELEVANCE_PROMPT")


@dataclass(frozen=True)
class SupervisorDomain:
    """How the supervisor prompts, and the text the supervisors build around them, name an env.

    GameBoy's values are GameBoyRL's own strings; its prompts are GameBoyRL's texts above and its
    [GAME] is filled with the game's name at render time, as GameBoyRL does."""
    env: str
    where: str = ""                   # "on an Android phone" (GENERIC_* only)
    subject: str = ""                 # what recorded knowledge is about (GENERIC_* only)
    controls: str = ""
    plan_example: str = ""
    regression_examples: str = ""
    stuck_example: str = ""
    hint_example: str = ""
    # Text the supervisors write around the prompts (_format.py, revising.py, info_subgoal.py).
    trace_heading: str = "what the player pressed, and why they said they pressed it:"
    trace_block_heading: str = "What they pressed, and the reason they gave for each:"
    regression_tail: str = "the player cannot finish this step from a state they have gone backwards into."
    relevance_frame_note: str = ("The images are: first the CURRENT screen the player is looking at, then the "
                                 "representative frame recorded with this entry.")
    relevance_no_frame_note: str = ("The image is the CURRENT screen the player is looking at. This entry has no "
                                    "recorded frame of its own — it was written from general knowledge of the game"
                                    " rather than from a playthrough, so judge it against its description and the "
                                    "current screen alone, and be correspondingly more willing to answer no.")
    #: The reply line each executor gives its reasons on (action_trace).
    reasoning_key: str = "Reasoning"
    #: Show a step the env rejected with its error in action lists (not GameBoy: GameBoyRL lists an
    #: unavailable action by name).
    show_step_failures: bool = True


GAMEBOY = SupervisorDomain(env="gameboy", show_step_failures=False)

_GENERIC_TEXT = dict(
    trace_heading="what the agent did, and why they said they did it:",
    trace_block_heading="What they did, and the reason they gave for each:",
    regression_tail="the agent cannot finish this step from a state they have gone backwards into.",
    relevance_frame_note=("The images are: first the CURRENT screen the agent is looking at, then the "
                          "representative frame recorded with this entry."),
    relevance_no_frame_note=("The image is the CURRENT screen the agent is looking at. This entry has no recorded "
                             "frame of its own — it was written from general knowledge rather than from a past "
                             "attempt, so judge it against its description and the current screen alone, and be "
                             "correspondingly more willing to answer no."))

ANDROID = SupervisorDomain(
    env="android", where="on an Android phone", subject="using an Android phone", reasoning_key="Reason",
    controls=("click (tap) an element, long_press an element, input_text into a field, scroll in a direction, "
              "navigate_back, navigate_home, open_app by name, keyboard_enter, wait"),
    plan_example=('Write "open the Contacts app until the contact list is showing" rather than "click [7]", and '
                  '"scroll the settings list until Display is visible" rather than "scroll down twice".'),
    regression_examples=("a dialog or menu that was open has closed, a field that was filled in is empty again, a "
                         "setting that was switched on is off, the agent has left the app or screen it had reached."),
    stuck_example="scroll down three times, then click element 12",
    hint_example=("scroll the contact list down until the name Ana appears, then click that name to open the "
                  "contact."),
    **_GENERIC_TEXT)

WEB = SupervisorDomain(
    env="web", where="in a web browser", subject="the websites an agent browses", reasoning_key="Thought",
    controls=("Click an element, Type into a field (which also submits it), Scroll the page or an area up or down, "
              "Wait, GoBack, Google (start a Google search), ANSWER (give the final answer)"),
    plan_example=('Write "search for the paper title in the search box at the top until a page of results appears" '
                  'rather than "Type [5]", and "scroll the results until the Filters panel is visible" rather than '
                  '"scroll down twice".'),
    regression_examples=("a filter that was applied has been cleared, a form field that was filled in is empty "
                         "again, a dropdown or panel that was open has closed, the agent has navigated away from a "
                         "page it had reached."),
    stuck_example="scroll down three times, then click [12]",
    hint_example=("click the search box at the top of the page and type the paper title; once the results page "
                  "appears, click the first result whose title matches."),
    **_GENERIC_TEXT)

DOMAINS = {"gameboy": GAMEBOY, "android": ANDROID, "web": WEB}


def prompts_for(*, domain: SupervisorDomain) -> dict:
    """{name: text} for every prompt in PROMPT_NAMES. GameBoy: GameBoyRL's texts unchanged ([GAME]
    left for the supervisor to fill). Android / Web: the GENERIC_* texts with the domain filled in."""
    module = globals()
    if domain.env == "gameboy":
        return {name: module[name] for name in PROMPT_NAMES}
    out = {}
    for name in PROMPT_NAMES:
        text = module["GENERIC_" + name]
        for key, value in (("[WHERE]", domain.where), ("[SUBJECT]", domain.subject), ("[CONTROLS]", domain.controls),
                           ("[PLAN_EXAMPLE]", domain.plan_example),
                           ("[REGRESSION_EXAMPLES]", domain.regression_examples),
                           ("[STUCK_EXAMPLE]", domain.stuck_example), ("[HINT_EXAMPLE]", domain.hint_example)):
            text = text.replace(key, value)
        out[name] = text
    return out
