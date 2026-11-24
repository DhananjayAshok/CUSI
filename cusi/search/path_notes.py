"""
Prompts and parsers of tree annotation (cusi.search.annotate). A chain starts at a node and is extended one edge
at a time; each call sees the chain's start screen, the screens before and after the new edge, what happened along
the chain so far (each edge's transition report, cusi.search.describe), the new edge's actions, and a sample of the
benchmark's train tasks (cusi.search.task_context).

- BUILD (no task yet): has the chain, from its start, now completed a task of the train tasks' kind? TOO_SHORT keeps
  extending (and names, on a Partial line, the best smaller task completed so far, or NONE); TASK names it.
- EXTEND (a task named): the new edge is CONTINUE (no effect on the task, or only opening further screens on its
  way: the end does not move), REVISE (on the reached screens the path changes something or reads out a specific
  answer, completing a fuller task that includes the current one) or NEW (it leaves for something unrelated: the task
  ended before this edge).
- GATE (after stage 2, one call per candidate): ACCEPT only if the path completes the task, the end screen shows it
  done or shows the answer, it reads like a train task (navigation-only only where the env's train flavours have
  it: GameBoy), and it is not a one-click wrapper; else REJECT with the failed criteria and a reason.

The prompts see only screens, transition reports and actions, never the envs' true state.
"""
import re
from cusi.search.context import action_line

TOO_SHORT, TASK, CONTINUE, REVISE, NEW = "TOO_SHORT", "TASK", "CONTINUE", "REVISE", "NEW"
#: Envs whose edges are long (5-19 steps): a task ends at the action that completes it inside the last edge, the
#: prompts read the edge's screen-by-screen report, and the fit is stricter (a checkable outcome, no navigation tacked
#: on, no exploration-label searches). GameBoy edges are 2-3 steps: its tasks end at the edge's end node.
STEP_ENVS = ("web", "android")
#: Whether only reaching a place is a task, per env, decided from the train flavours (configs/annotation/):
#: GameBoy's navigation and menu flavours are just that ("Enter the Pokemon Center", "Open the Map"); no Web or Android
#: train task is (the Web ones that open a page go on to ask for an answer there).
REACH_NOTES = {
    "gameboy": " Reaching a place or a menu (entering a building, leaving a room or cave, arriving next to someone or "
               "something named, opening a game menu) also counts, as in the examples.",
    "web": " Only opening or reaching a page, section, tab, menu or article is not a task: it counts once something is "
           "done there or a specific answer is read off it.",
    "android": " Only opening or reaching a screen, settings page, menu, dialog or details view is not a task: it "
               "counts once something is changed there or a specific answer is read off it.",
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
is smaller than a full task: it must still meet every rule above but the size of the goal (achieved, specific about \
exactly which setting, item, page or place, named from the screens, no accidental or fragmentary typed text such as \
a search for "ol"), or NONE if nothing meets them.

Reply in exactly this format, each task as one sentence on its line, without any [label]:
Verdict: {too_short} or {task}
Task: <the task, only when the verdict is {task}>
Partial: <only when the verdict is {too_short}: the best task completed so far, or NONE>"""

EXTEND_PROMPT = _HEAD + """
Before the latest transition, the sequence had completed this task (image 2 shows it done):
"{current}"

Question: what is the latest transition to that task? Choose one:
- {cont}: the latest transition has no effect on the task, only repeats or confirms it, or only opens further \
screens on the way of it (an ineffective action, a scroll, a step back that keeps the result, opening a submenu); \
"{current}" still names what the whole sequence achieved, and its end does not move.
- {revise}: on the screens it reached, the path now changes something (turns a toggle or setting, enters or saves \
a form field, sorts or filters, adds or deletes an item, takes a purchase or booking step; in a game, talks to \
someone, reads a sign or text, battles, uses an item) or reads out a specific answer, so that the whole sequence, \
image 1 to image 3, completes a fuller task that includes "{current}". Give that task, under the same rules as any \
task: of the kind in the examples, achieved, naming exactly which setting, item, value or page, and no accidental or \
fragmentary typed text. Opening another screen, page, menu, dialog, tab or article is never a revision: it is \
{cont} while it stays on the way of the same task, {new} otherwise.
- {new}: the latest transition leaves for something unrelated (another app, site or area not on the way of the \
task), starts an unrelated goal, or undoes the task; the task ended at image 2.

Reply in exactly this format, the task as one sentence on the Task line, without any [label]:
Verdict: {cont}, {revise} or {new}
Task: <the revised task, only when the verdict is {revise}>"""

ACCEPT, REJECT = "ACCEPT", "REJECT"
#: The fit gate's criteria, named by letter in its reply.
GATE_CRITERIA = {"a": "the path actually completes the task", "b": "the end state shows it done or shows the answer",
                 "c": "it reads like a train task of its kind", "d": "it is not a trivial one-click wrapper"}

GATE_PROMPT = """You are checking a task proposed for recorded interaction with {env_description}.{app_note} Image 1 \
is the screen the recording starts from; image 2 the screen it ends on.

What happened, transition by transition:
{history}

Proposed task: "{task}"

Users of this environment are given tasks like these:
{train}

Accept the task only if ALL of these hold:
(a) the recording actually completes it: every part of it was done in these transitions, not only reached or begun \
(reaching a form is not filling it in; opening a page is not finding something on it);
(b) image 2 shows it done, or shows the specific answer it asks for;
(c) it reads like the example tasks: a specific goal of the same kind and form{nav_rule};
(d) it is not a trivial wrapper around a single click or keystroke.

Reply in exactly this format:
Verdict: {accept} or {reject}
Failed: <the letters of the criteria it fails, such as "a, c", or none>
Reason: <one sentence>"""

#: Navigation-only tasks fit the examples only where the train flavours have them (the env's REACH_NOTES decision).
GATE_NAV_RULES = {"gameboy": "; reaching a named place or opening a game menu is a goal of that kind here",
                  "web": "; only going to a page, section or article is not a goal of that kind here",
                  "android": "; only opening a screen, settings page, menu or dialog is not a goal of that kind here"}


def gate_prompt(*, env_name: str, env_description: str, app_note: str, reports: list, task: str, train: str,
                n_last: int = 2) -> str:
    """Two images (start and end screens); in STEP_ENVS the last edge's screens up to the completing action, image
    n_last being the screen right after it."""
    fields = dict(env_description=env_description, app_note=app_note, history=history_block(reports=reports),
                  task=task, train=train, nav_rule=GATE_NAV_RULES[env_name], accept=ACCEPT, reject=REJECT)
    if env_name in STEP_ENVS:
        return GATE_PROMPT_STEPS.format(**fields, n_last=n_last)
    return GATE_PROMPT.format(**fields)


def parse_gate(*, text: str) -> dict:
    """{"verdict" (ACCEPT / REJECT / None when unreadable), "failed" (criterion letters), "reason"}."""
    text = str(text or "")
    m = re.search(r"verdict\s*:\s*\**\s*(ACCEPT|REJECT)", text, re.IGNORECASE)
    f = re.search(r"failed\s*:\s*(.*)", text, re.IGNORECASE)
    r = re.search(r"reason\s*:\s*(.*)", text, re.IGNORECASE)
    failed = sorted(set(re.findall(r"\b([abcd])\b", f.group(1).lower()))) if f else []
    return {"verdict": m.group(1).upper() if m else None, "failed": failed,
            "reason": " ".join(r.group(1).split()) if r else ""}


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
                 train: str, windows=None) -> str:
    """In STEP_ENVS the prompt also shows the latest transition's windows and asks for a Step line."""
    fields = dict(env_description=env_description, app_note=app_note, history=history_block(reports=reports),
                  n_latest=len(steps), actions=actions_block(steps=steps),
                  latest=f"What it did: {latest or '(no report)'}", train=train, too_short=TOO_SHORT, task=TASK,
                  reach_note=REACH_NOTES[env_name])
    if env_name in STEP_ENVS:
        return BUILD_PROMPT_STEPS.format(**fields, windows_block=windows_block(windows=windows))
    return BUILD_PROMPT.format(**fields)


def extend_prompt(*, env_description: str, app_note: str, reports: list, steps: list, latest: str, train: str,
                  current: str, env_name: str = "gameboy", windows=None) -> str:
    fields = dict(env_description=env_description, app_note=app_note, history=history_block(reports=reports),
                  n_latest=len(steps), actions=actions_block(steps=steps),
                  latest=f"What it did: {latest or '(no report)'}", train=train, current=current, cont=CONTINUE,
                  revise=REVISE, new=NEW)
    if env_name in STEP_ENVS:
        return EXTEND_PROMPT_STEPS.format(**fields, windows_block=windows_block(windows=windows))
    return EXTEND_PROMPT.format(**fields)


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
        if re.match(r"\s*(task|partial|verdict|step)\s*:", x, re.IGNORECASE):
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


# --------------------------------------------------------------------------- step-level envs (STEP_ENVS)
#: The latest transition's screen-by-screen report: the stage-1 windows name its screens "Frame k", frame 1 being the
#: screen before the transition and frame k + 1 the screen right after its action k.
STEPS_WINDOWS = """
Screen by screen (frame 1 is the screen before the latest transition, frame k+1 the screen right after its action k):
{windows}"""

STEP_LINE = """Step: <the number of the action in the latest transition right after which the task (or the \
partial task) was complete, from the actions and frames above>"""

STEPS_BUILD_EXTRA = """- it has a checkable outcome: a change the screen shows, or a specific answer (a name, number, date, \
price, title, status) it asks for. "Find information about X", "search for information on X" or "look up X" with \
no specific question is no task;
- a search counts only with a query a user would type for their own purpose: a label that describes the app or an \
activity ("Playlist management", "Map exploration", "Search functionality") is not such a query;
"""

STEPS_EXTEND_EXTRA = """ Adding "and view", "and open", "and go to" or "and check" something after the task was done is \
never a revision either: that is {cont} or {new}, and the shorter task stands."""

GATE_PROMPT_STEPS = GATE_PROMPT.replace(
    """Image 1 \
is the screen the recording starts from; image 2 the screen it ends on.""",
    """Image 1 \
is the screen the recording starts from; images 2 to {n_last} are the screens of its last transition up to the action \
that is said to complete the task, image {n_last} being the screen right after that action.""").replace(
    """(b) image 2 shows it done, or shows the specific answer it asks for;
(c) it reads like the example tasks: a specific goal of the same kind and form{nav_rule};""",
    """(b) image {n_last} shows it done, or shows the specific answer it asks for (the last transition's report may \
go on past that action: judge the task by the screens up to image {n_last});
(c) it reads like the example tasks: a specific goal of the same kind and form{nav_rule}; it has a checkable outcome \
(a change the screen shows, or a specific answer such as a name, number, date, price, title or status), so "Find \
information about X" or "search for information on X" with no specific question fails; it does not tack navigation \
onto a finished task ("... and view the images of ...", "... and view the details of ..."); a search query that only \
labels the app or an activity ("Playlist management", "Map exploration") fails;""")


def _steps_build(text: str) -> str:
    text = text.replace("{latest}\n", "{latest}{windows_block}\n")
    text = text.replace("- it names only the goal:", STEPS_BUILD_EXTRA + "- it names only the goal:")
    return text.replace("""Partial: <only when the verdict is {too_short}: the best task completed so far, or NONE>""",
                        """Partial: <only when the verdict is {too_short}: the best task completed so far, or NONE>
""" + STEP_LINE)


def _steps_extend(text: str) -> str:
    text = text.replace("{latest}\n", "{latest}{windows_block}\n")
    text = text.replace("""{cont} while it stays on the way of the same task, {new} otherwise.""",
                        """{cont} while it stays on the way of the same task, {new} otherwise.""" + STEPS_EXTEND_EXTRA)
    return text.replace("""Task: <the revised task, only when the verdict is {revise}>""",
                        """Task: <the revised task, only when the verdict is {revise}>
""" + STEP_LINE)


BUILD_PROMPT_STEPS, EXTEND_PROMPT_STEPS = _steps_build(BUILD_PROMPT), _steps_extend(EXTEND_PROMPT)


def windows_block(*, windows) -> str:
    """The stage-1 window reports of the latest transition, or nothing when there are none."""
    lines = [f"- frames {w.get('frames')}: {' '.join(str(w.get('text') or '').split())}" for w in windows or []
             if isinstance(w, dict)]
    return STEPS_WINDOWS.format(windows="\n".join(lines)) if lines else ""


def parse_step(*, text: str, n_steps: int) -> int:
    """The completing action's number from a reply's Step line, clipped to 1..n_steps; n_steps (the edge's end)
    when there is none."""
    m = re.search(r"\bstep\s*:\s*\**\s*(\d+)", str(text or ""), re.IGNORECASE)
    return min(max(int(m.group(1)), 1), n_steps) if m else n_steps
