"""
Notes for the practice and knowledge pipelines, written by `cusi.search.annotate` on a finished tree:

- per node (NodeNotes): a description of its screen, for the knowledge document, and the tasks that could be done
  from it (source "node");
- per edge (EdgeNotes, the segment from a node's parent to it): the task the transition completed (hindsight; source
  "edge", saved on the parent, the node the task starts from) and the actionable knowledge it shows.

The prompts see only screens, screen text and actions, never the envs' true state.
"""
import re
from cusi.search.context import action_line
from cusi.search.describe import parse_bullets
from cusi.search.features import screen_text

NO_TASK = "NO TASK"

NODE_PROMPT = """You are looking at a screen of {env_description}.

Text currently on the screen:
{texts}

First, describe this screen for someone who will later use this environment without seeing it: which screen, page, \
menu or place this is, what it shows, and what can be done from it (its buttons, options, links, exits or other \
interactive elements), naming them as they appear.

Then propose tasks that a user could set out to accomplish starting from this exact screen. Write each task as the \
goal a person would be given, not as instructions for which buttons to press: say what to achieve or what to find \
out, and leave how to the person. Good tasks:
- take several actions to complete, not a single click or button press;
- are grounded in what this screen shows or clearly leads to, and name their specific targets (when something \
has to be entered or chosen, say exactly what: which city, which name, which value);
- end in a state that is visible on the screen, or ask for a piece of information that can be read off it;
- can be done without information the user does not have (such as passwords or personal details).
Cover the different things this screen offers rather than listing near-identical variations (one task about a kind \
of item, not one task per item in a list).

{task_examples}

Reply in exactly this format:
Description: <three to five sentences>
Tasks:
- <task>
- <task>"""

EDGE_PROMPT = """These two screenshots show {env_description}: the first is a starting screen, the second the screen \
reached from it after the {n_actions} actions below.

Actions, in order:
{actions}

What happened along the way:
{description}

1. Task: what did this sequence achieve? State it as the goal a person starting from the first screen would be \
given to reach exactly this outcome: say what to achieve or find out, not which buttons were pressed, and name the \
specific target (page, menu, setting, information, place or item) so that it can be checked from the final screen. \
If the sequence achieved nothing meaningful (it only moved around without reaching anything new, ended where it \
started, or nothing changed), or its outcome takes a single action to reach, write {no_task}.
{task_examples}
2. Knowledge: list the facts this sequence shows that would help someone complete future tasks in this environment: \
how to get from one screen to another, what a control or action does, where something is, or what to avoid. Each fact \
must be borne out by these screens, specific and actionable, name screens and controls as they appear on the screen \
(never by numeric labels or indices), and make sense on its own. Leave the list empty if there are none.

Reply in exactly this format:
Task: <task or {no_task}>
Knowledge:
- <fact>
- <fact>"""

#: Few-shot task examples per env, written in the style of the env's benchmark tasks (their length, terseness and
#: answer formats) but set in invented environments, so that no benchmark content reaches the prompts.
TASK_EXAMPLES = {
    "gameboy": """Write tasks as short commands with a single goal, the way a player would be told what to do next; \
they may refer to things by where they are relative to the player. For example, tasks of this kind in another game \
(a farming game, not this one) would be:
- "Talk to the mayor"
- "Buy turnip seeds"
- "Read the notice board to your left"
- "Water the crops below you\"""",
    "web": """Write tasks as requests a user would type: name the concrete conditions the result must meet (dates, \
counts, filters, values), and when the task asks for information, say what to report. For example, tasks of this \
kind on other websites (not this one) would be:
- On a public library's website: "Find a novel by a Portuguese author published after 2015 that is available as an \
e-book, and give its title and page count."
- On a recipe blog: "List 3 soup recipes that take under 30 minutes to make."
- On a weather site: "What is the forecast high in Lisbon for tomorrow?\"""",
    "android": """Write tasks as instructions a phone's owner would give: name the app, give the exact values to \
enter or choose, and when the task asks for information, say what form the answer should take. For example, tasks of \
this kind with other apps (not on this phone) would be:
- "In the Notebook app, create a note titled 'Groceries' containing 'eggs, milk, rice'."
- "How many alarms are set in the Alarmist app? Answer with just the number."
- "Turn on dark mode in the Weatherly app, then add Lisbon to its saved locations.\"""",
}

_DESCRIPTION_RE = re.compile(r"description\s*:\s*(.*?)(?=\n\s*tasks\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_TASKS_RE = re.compile(r"\n\s*tasks\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_TASK_RE = re.compile(r"task\s*:\s*(.*?)(?=\n\s*knowledge\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_KNOWLEDGE_RE = re.compile(r"knowledge\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
#: Bullets are kept whole (parse_bullets cuts at its own default otherwise).
MAX_ITEM_CHARS = 500


def parse_node(*, text: str) -> tuple:
    """(description or None, tasks) from a 'Description: ... Tasks: - ...' reply."""
    text = str(text or "")
    m = _DESCRIPTION_RE.search(text)
    description = " ".join(m.group(1).split()) if m else None
    t = _TASKS_RE.search(text)
    return description or None, parse_bullets(block=t.group(1), max_chars=MAX_ITEM_CHARS) if t else []


def parse_edge(*, text: str) -> tuple:
    """(task or None, knowledge, parsed) from a 'Task: ... Knowledge: - ...' reply; NO TASK is None."""
    text = str(text or "")
    m = _TASK_RE.search(text)
    task = " ".join(m.group(1).split()) if m else ""
    k = _KNOWLEDGE_RE.search(text)
    knowledge = parse_bullets(block=k.group(1), max_chars=MAX_ITEM_CHARS) if k else []
    if task.rstrip(".").upper() == NO_TASK:
        task = ""
    return task or None, knowledge, bool(m)


class NodeNotes:
    def __init__(self, *, vlm, task_examples: str, max_new_tokens: int = 1000, temperature: float = 0.0) -> None:
        """task_examples: the env's TASK_EXAMPLES entry."""
        self.vlm, self.task_examples = vlm, task_examples
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def __call__(self, *, env_description: str, frame, text_boxes: list) -> dict:
        """{"screen_description", "node_tasks", "node_notes_raw"}; frame is the node's unlabelled screen (PIL)."""
        prompt = NODE_PROMPT.format(env_description=env_description, texts=screen_text(boxes=text_boxes),
                                    task_examples=self.task_examples)
        out = str(self.vlm.infer(texts=prompt, images=[frame], max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature)["output"])
        description, tasks = parse_node(text=out)
        return {"screen_description": description, "node_tasks": tasks, "node_notes_raw": out}


class EdgeNotes:
    def __init__(self, *, vlm, task_examples: str, max_new_tokens: int = 1000, temperature: float = 0.0) -> None:
        """task_examples: the env's TASK_EXAMPLES entry."""
        self.vlm, self.task_examples = vlm, task_examples
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def __call__(self, *, env_description: str, start_frame, end_frame, steps: list,
                 description: str) -> dict:
        """{"edge_task" (None: no task), "knowledge", "edge_notes_raw"}: start_frame is the parent's screen, end_frame
        the node's (PIL); steps the segment's steps; description its transition report."""
        actions = "\n".join(f"{i + 1}. {action_line(action=s.action, valid=s.info.get('valid', True))}"
                            for i, s in enumerate(steps))
        prompt = EDGE_PROMPT.format(env_description=env_description, n_actions=len(steps), actions=actions,
                                    description=description or "(no description)", no_task=NO_TASK,
                                    task_examples=self.task_examples)
        out = str(self.vlm.infer(texts=prompt, images=[start_frame, end_frame], max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature)["output"])
        task, knowledge, _ = parse_edge(text=out)
        return {"edge_task": task, "knowledge": knowledge, "edge_notes_raw": out}
