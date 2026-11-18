"""
Prompts and parsers of tree annotation (cusi.search.annotate). A chain starts at a node and is extended one edge
at a time; each call sees the chain's start screen, the screens before and after the new edge, what happened along
the chain so far (each edge's transition report, cusi.search.describe), the new edge's actions, and a sample of the
benchmark's train tasks (cusi.search.task_context).

- BUILD (no task yet): has the chain, from its start, now completed a task of the train tasks' kind? TOO_SHORT keeps
  extending (and names, on a Partial line, the best smaller task completed so far, or NONE); TASK names it.
- EXTEND (a task named): the new edge is CONTINUE (more of the same task, or no effect on it), REVISE (the chain now
  completes a larger or more specific task that includes the current one) or NEW (something different starts: the
  task ended before this edge).

The prompts see only screens, transition reports and actions, never the envs' true state.
"""
import re
from cusi.search.context import action_line

TOO_SHORT, TASK, CONTINUE, REVISE, NEW = "TOO_SHORT", "TASK", "CONTINUE", "REVISE", "NEW"
#: Recorded in a run's annotation meta: cached verdicts of one prompt version are never mixed with another's.
PROMPT_VERSION = 5
#: Reaching a place is a task in every env (p5, user's call), when the place is specific and named.
REACH_NOTES = {
    "gameboy": " Reaching a place (entering a building, leaving a room or cave, arriving next to someone or something "
               "named) also counts, as in the examples.",
    "web": " Reaching a specific, named page, section or view (a product page, a documentation page, a filtered list "
           "of results) also counts: write it as a goal such as \"Go to the ... page on ...\".",
    "android": " Reaching a specific, named screen, settings page or dialog also counts: write it as a goal such as "
               "\"Open the ... settings in ...\".",
}

_HEAD = """You are labelling recorded interaction with {env_description}.{app_note} The images, in order:
image 1 is the screen the sequence starts from; image 2 the screen just before the latest transition; image 3 the \
screen after it.

What happened along the sequence, transition by transition from the starting screen:
{history}

The latest transition ({n_latest} actions):
{actions}
{latest}

Users of this environment are given tasks like these (examples from other screens of it; the tasks you write need \
not be these, but should be of this kind and written in this form):
{train}
"""

BUILD_PROMPT = _HEAD + """
Question: taken as a whole, from image 1 to image 3, has this sequence completed a task of the kind in the examples? \
A task counts only if all of these hold:
- it is a goal of the same kind as the examples: it changes something (a setting, an item created, edited, moved or \
deleted, a value entered, a choice made) or it asks a question whose answer image 3 shows.{reach_note}
- these transitions actually achieved it: image 3 shows it done, or shows the answer;
- it is more than a lone scroll, keystroke, or failed or ineffective action;
- it is specific: it names exactly which setting, item, value, page, place or person (never "a switch", "an option", \
"a specific location"); if the screens do not show which, it is no task;
- everything it names appears in the screens or the reports, and is something a user would plausibly ask for: random, \
garbled or fragmentary text typed by accident (such as "olz, l" or a search for "be") makes no task;
- it names only the goal: leave out steps after the goal was reached (such as opening another menu afterwards).
If not, answer {too_short}: the sequence so far is only part of a task, wandering, or nothing. The sequence will be \
extended and asked again. Then, on the Partial line, give the best task the sequence has completed so far even if it \
is smaller than these rules ask for (still specific, achieved and named from the screens), or NONE if it has \
completed nothing at all.

Reply in exactly this format, each task as one sentence on its line, without any [label]:
Verdict: {too_short} or {task}
Task: <the task, only when the verdict is {task}>
Partial: <only when the verdict is {too_short}: the best task completed so far, or NONE>"""

EXTEND_PROMPT = _HEAD + """
Before the latest transition, the sequence had completed this task (image 2 shows it done):
"{current}"

Question: what is the latest transition to that task? Choose one:
- {cont}: the latest transition has no effect on the task or only repeats or confirms it (an ineffective action, a \
scroll on the same screen, a step back that keeps the result); "{current}" still names what the whole sequence \
achieved.
- {revise}: the latest transition advances the same goal further, so that the whole sequence, image 1 to image 3, \
now completes a more complete or more specific version of it (a further condition of the same goal met, the change \
finished, the answer found). Give that task. Moving on to a different screen, setting, item or question is not a \
revision.
- {new}: the latest transition does something different or unrelated, starts another goal, or undoes the task; the \
task ended at image 2.

Reply in exactly this format, the task as one sentence on the Task line, without any [label]:
Verdict: {cont}, {revise} or {new}
Task: <the revised task, only when the verdict is {revise}>"""

_VERDICT_RE = re.compile(r"verdict\s*:\s*\**\s*([A-Z_]+)", re.IGNORECASE)
#: How many earlier transitions are written out in full; the ones before are listed by their first sentence only.
FULL_HISTORY = 6


def app_note(*, app: str, where: str) -> str:
    """A sentence naming the app on a screen, or nothing."""
    return f' {where} is in the {app} app: call it "{app}" when a task names the app.' if app else ""


def history_block(*, reports: list) -> str:
    """The chain's transition reports, oldest first, numbered; earlier than the last FULL_HISTORY, first sentences."""
    if not reports:
        return "(none: the latest transition is the first)"
    lines = []
    for i, r in enumerate(reports):
        text = " ".join(str(r or "(no report)").split())
        if i < len(reports) - FULL_HISTORY:
            text = text.split(". ")[0].rstrip(".") + "."
        lines.append(f"{i + 1}. {text}")
    return "\n".join(lines)


def actions_block(*, steps: list) -> str:
    return "\n".join(f"{i + 1}. {action_line(action=s.action, valid=s.info.get('valid', True))}"
                     for i, s in enumerate(steps))


def build_prompt(*, env_name: str, env_description: str, app_note: str, reports: list, steps: list, latest: str,
                 train: str) -> str:
    return BUILD_PROMPT.format(env_description=env_description, app_note=app_note,
                               history=history_block(reports=reports), n_latest=len(steps),
                               actions=actions_block(steps=steps), latest=f"What it did: {latest or '(no report)'}",
                               train=train, too_short=TOO_SHORT, task=TASK, reach_note=REACH_NOTES[env_name])


def extend_prompt(*, env_description: str, app_note: str, reports: list, steps: list, latest: str, train: str,
                  current: str) -> str:
    return EXTEND_PROMPT.format(env_description=env_description, app_note=app_note,
                                history=history_block(reports=reports), n_latest=len(steps),
                                actions=actions_block(steps=steps), latest=f"What it did: {latest or '(no report)'}",
                                train=train, current=current, cont=CONTINUE, revise=REVISE, new=NEW)


def clean_task(*, text: str, field: str = "task") -> str:
    """The task after '<field>:' (Task or Partial), as one line: a leading [flavour] label and list dashes (the
    examples' layout, which replies copy: 'Task: [Settings]\\n- Open ...') are dropped; the next field's line ends
    it."""
    m = re.search(rf"\b{field}\s*:(.*)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    body = re.sub(r"^\s*\[[^\]\n]*\]\s*", "", m.group(1).strip())
    lines = []
    for x in body.splitlines():
        if re.match(r"\s*(task|partial|verdict)\s*:", x, re.IGNORECASE):
            break
        if x.strip():
            lines.append(re.sub(r"^\s*[-*]\s*", "", x).strip())
    return " ".join(" ".join(lines[:1]).split()).strip().strip('"').strip()


def parse_partial(*, text: str) -> str:
    """A TOO_SHORT reply's best task completed so far, or None (NONE, a placeholder or no Partial line)."""
    task = clean_task(text=str(text or ""), field="partial")
    if not task or task.upper().startswith(("<", "NONE", "N/A")):
        return None
    return task


def parse_verdict(*, text: str, allowed: tuple) -> tuple:
    """(verdict, task or None, parsed): the first allowed verdict word; a verdict that needs a task but has none is
    not parsed."""
    text = str(text or "")
    m = _VERDICT_RE.search(text)
    verdict = m.group(1).upper() if m else None
    if verdict not in allowed:
        return None, None, False
    task = clean_task(text=text[m.end():])
    if verdict in (TASK, REVISE):
        if not task or task.upper().startswith(("<", "NONE", "N/A")):
            return None, None, False
        return verdict, task, True
    return verdict, None, True
