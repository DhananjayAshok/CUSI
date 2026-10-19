"""
The GameBoy subgoal supervisor's prompts, GameBoyRL's texts verbatim: the plan, the frame-window summary, the
step judge, the regression check, the resume hint and the plan review.
"""

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
