"""
AndroidWorld's M3A agent with the final Android design (decisions.md E6.9, E8.0): M3A's prompts and step handling,
plus a carried notes field (F27) and Minitap's pieces F55–F61, all fixed behaviour. The two env-side pieces (F57
typing without Enter, F60 resource ids) live in AndroidPlayEnv. M3A's prompt text comes from the android_world
submodule and is patched here; every patch's anchor is checked at import, so a submodule change fails loudly.
"""
import json
import re
from typing import Optional
import numpy as np
from android_world.agents import agent_utils, m3a, m3a_utils
from android_world.env import adb_utils, json_action
from cusi.envs.android_world import MSG_EXEC_FAILED, describe_actions, patch
from cusi.agents.executors.base import EXPLORE_GOAL, GUIDANCE_END, GUIDANCE_START, SECRET_NOTE, Decision, Executor

# M3A.step's history messages, verbatim.
MSG_BAD_FORMAT = "Output for action selection is not in the correct format, so no action is performed."
MSG_BAD_ACTION = ("Can not parse the output to a valid action. Please make sure to pick the action from the list"
                  " with required parameters (if any) in the correct JSON format!")
MSG_NO_FINISH = "The status and answer actions are not available while exploring; keep exploring the phone."
FINISH_ACTIONS = ("status", "answer")

# F27: notes the model writes in each action reply and sees again next step (M3A's history keeps one-line summaries).
NOTES_RULE = ("Notes: <facts you have found so far that the task needs (values, names, what you have checked),"
              " plus what you tried that failed; keep it up to date>")
NOTES_SHOWN = "Your notes so far (written by you on earlier steps): "
NOTES_ASK = "In your answer, after the Reason line and before the Action line, add these lines:\n"
NOTES_MAX_CHARS = 800
_FIELD_STOP = r"(?:\n\s*(?:Action|Notes|Plan|Reason|Reasoning|Thought)\s*:|$)"

# F55: the guidance tip for clear_text (the action text itself, and F57's, is the env's describe_actions).
DELETE_TIP_OLD = ("Sometimes there is some default text in the text field you want to type in, remember to delete them"
                  " before typing.")
DELETE_TIP_NEW = ("If the field already contains text that should not stay (default text, an old value), set"
                  " \"clear_text\": true instead of deleting it by hand.")
# F58: Minitap's Cortex critical rules.
MINITAP_RULES = (
    "Additional rules:\n"
    "- Before deciding, review the history: if an action already failed or did nothing, do NOT repeat it unchanged;"
    " understand why and change strategy (ask how a human would do it differently).\n"
    "- Only finish (status complete) on OBSERVED evidence: the screen must show that every part of the goal is done."
    " If the goal has formatting constraints (exact text, new lines, order), check the result on screen and fix it"
    " before finishing.\n"
    "- Data fidelity: type and report content EXACTLY as given in the goal or shown on screen; do not correct,"
    " reformat, translate or add to it unless asked.\n"
    "- Forms: before concluding a field is missing, scroll through the whole form; never put data into the wrong field"
    " if the right one was seen earlier.\n")
# F59: one planner call writes subgoals; the action prompt shows them as a checklist the model ticks.
PLANNER_PROMPT = (
    "You are the planner for an agent operating an Android phone. Break the user's goal into 2 to 6 sequential"
    " subgoals: purpose-driven milestones, not button-by-button steps; no loops (write repeated work as separate"
    " subgoals). If the goal has formatting or exact-content constraints, end with a subgoal to verify the result on"
    " screen and fix it. Prefer opening apps with open_app.\n\nThe goal: {goal}\n\nThe current screen is attached.\n"
    'Answer with JSON only: {{"subgoals": ["...", "..."]}}')
MAX_SUBGOALS = 8
SUBGOALS_HEADER = "Subgoal plan ([x] = completed, seen on screen):\n"
SUBGOALS_ASK = ("In your answer, after the Reason line and before the Action line, add a line \"Completed subgoals:"
                " <numbers of the subgoals now completed, as observed on the screen, or none>\". Finish the task only"
                " when every subgoal is completed.")
# F56: what a text field holds after typing.
READBACK_MATCH_CHARS = 40


def insert(*, prompt: str, text: str) -> str:
    """text before M3A's closing 'Your Answer:' line."""
    i = prompt.rfind("Your Answer:")
    if i < 0:
        raise ValueError("M3A prompt anchor 'Your Answer:' not found (android_world changed?)")
    return prompt[:i] + text.strip() + "\n\n" + prompt[i:]


def drop_finish_actions(*, prompt: str) -> str:
    """The action list without the status and answer actions (no task: nothing to finish or answer)."""
    lines = prompt.split("\n")
    kept = [l for l in lines if not (l.startswith("- ") and any(f'"action_type": "{a}"' in l for a in FINISH_ACTIONS))]
    if len(lines) - len(kept) != 3:
        raise ValueError("M3A prompt: expected 3 finishing action lines (android_world changed?)")
    return "\n".join(kept)


def read_notes(*, response: str) -> str:
    """The Notes field of an action reply, whitespace-collapsed and capped; '' if absent."""
    m = re.search(rf"Notes\s*:\s*(.*?){_FIELD_STOP}", response or "", re.IGNORECASE | re.DOTALL)
    return " ".join(m.group(1).split())[:NOTES_MAX_CHARS] if m else ""


def notes_block(*, notes: str) -> str:
    return (f"{NOTES_SHOWN}{notes}\n" if notes else "") + NOTES_ASK + NOTES_RULE


def subgoals_block(*, subgoals: list, done: set) -> str:
    plan = "\n".join(f"{i + 1}. [{'x' if i in done else ' '}] {s}" for i, s in enumerate(subgoals))
    return SUBGOALS_HEADER + plan + "\n" + SUBGOALS_ASK


def parse_subgoals(*, response: str) -> list:
    m = re.search(r"\{.*\}", response or "", re.DOTALL)
    try:
        return [str(s) for s in json.loads(m.group(0))["subgoals"]][:MAX_SUBGOALS] if m else []
    except Exception:
        return []


def completed_subgoals(*, response: str, n: int) -> set:
    m = re.search(r"Completed subgoals\s*:\s*([^\n]*)", response or "", re.IGNORECASE)
    return {int(k) - 1 for k in re.findall(r"\d+", m.group(1)) if 1 <= int(k) <= n} if m else set()


def device_context(*, env) -> str:
    """F61: the foreground app and the device date/time (one adb call)."""
    aw = getattr(env, "_env", None)
    app, when = "unknown", "unknown"
    try:
        app = aw.foreground_activity_name.split("/")[0]
    except Exception:
        pass
    try:
        r = adb_utils.issue_generic_request(["shell", "date", "+%A_%Y-%m-%d_%H:%M"], aw.controller)
        when = r.generic.output.decode().strip().replace("_", " ") or when
    except Exception:
        pass
    return f"Device context: foreground app {app}; device date and time {when}."


def _parse_elements(text: str) -> list:
    """M3A element-list lines -> [(index, text, is_editable)]."""
    out = []
    for m in re.finditer(r'UI element (\d+): \{(.*)\}', text or ""):
        body = m.group(2)
        t = re.search(r'"text": "(.*?)", "', body)
        out.append((int(m.group(1)), t.group(1) if t else "", '"is_editable": True' in body))
    return out


def readback_note(*, action: str, ui_elements: str) -> Optional[str]:
    """F56: after an input_text, what the editable fields now hold; None for any other action."""
    try:
        parsed = json.loads(action) if isinstance(action, str) else action
    except Exception:
        try:
            parsed = agent_utils.extract_json(action)
        except Exception:
            parsed = None
    if not parsed or parsed.get("action_type") != "input_text" or not parsed.get("text"):
        return None
    typed = str(parsed["text"])
    fields = [t for _, t, editable in _parse_elements(ui_elements) if editable]
    hit = [t for t in fields if typed.strip() and typed.strip()[:READBACK_MATCH_CHARS] in t]
    if hit:
        return f"[Field check] After typing, the text field contains: {json.dumps(hit[0][:300])}"
    if fields:
        shown = "; ".join(json.dumps(t[:120]) for t in fields[:3])
        return (f"[Field check] The typed text {json.dumps(typed[:80])} is not in any text field on the screen now"
                f" (editable fields show: {shown}). Check whether it was entered where you intended.")
    return (f"[Field check] No text field is on the screen after typing {json.dumps(typed[:80])}; check that it was"
            " entered and saved where you intended.")


def action_prompt(*, goal: Optional[str], history: list, ui_elements: str, guidance: Optional[str], notes: str,
                  subgoals: list, done: set, context: str) -> str:
    """M3A's action prompt with the final design; goal None = exploring (no finishing actions, no plan).
    Blocks are inserted in the confirmed order: notes, rules, subgoal plan, device context."""
    if guidance:
        text = f"{guidance}\n{SECRET_NOTE}"
        prompt = m3a._action_selection_prompt(goal or EXPLORE_GOAL, history, ui_elements, [text])
        block = f"For The Current Task:\n- {text}\n"
        assert prompt.count(block) == 1
        prompt = prompt.replace(block, GUIDANCE_START + block + GUIDANCE_END)
    else:
        prompt = m3a._action_selection_prompt(goal or EXPLORE_GOAL, history, ui_elements, None)
    if goal is None:
        prompt = drop_finish_actions(prompt=prompt)
    prompt = insert(prompt=prompt, text=notes_block(notes=notes))                        # F27
    prompt = describe_actions(prompt=prompt)                                             # F55, F57
    prompt = patch(prompt=prompt, old=DELETE_TIP_OLD, new=DELETE_TIP_NEW, flat=True)      # F55
    prompt = insert(prompt=prompt, text=MINITAP_RULES)                                    # F58
    if subgoals:
        prompt = insert(prompt=prompt, text=subgoals_block(subgoals=subgoals, done=done))  # F59
    return insert(prompt=prompt, text=context)                                           # F61


# Every anchor, checked once at import, with and without a task.
for _goal in ("g", None):
    action_prompt(goal=_goal, history=[], ui_elements="", guidance=None, notes="", subgoals=[], done=set(), context="")


class M3AExecutor(Executor):
    """M3A with the final Android design, on an AndroidPlayEnv built with som=True and stabilize_after_action=False
    (each decision re-observes the screen, as M3A does). task None = exploring: no finishing, no planner call."""

    name = "m3a"
    env_name = "android"
    MAX_NEW_TOKENS = 1000      # benchmark_adapters/android_world_adapter.py InferenceModelWrapper
    TEMPERATURE = 0.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._notes = ""
        self._subgoals: list = []
        self._done: set = set()
        self._planned = False
        self._summaries: list = []
        self._pending: Optional[dict] = None
        self._before: Optional[tuple] = None

    def reset_memory(self) -> None:
        self._notes = ""
        self._subgoals, self._done, self._planned = [], set(), False
        self._summaries = []
        self._pending = None
        self._before = None

    def history(self) -> list:
        """M3A's step summaries; numbering continues in the next leg."""
        return list(self._summaries)

    def restore_history(self, history: list) -> None:
        self._summaries = list(history)

    def _history(self) -> list:
        return ["Step " + str(i + 1) + "- " + s for i, s in enumerate(self._summaries)]

    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        # Error feedback reaches M3A through its history, not `error`.
        if self.task is not None and not self._planned:   # F59, on the observation handed in (as confirmed)
            self._planned = True
            out = self.call(tag="plan", prompt=PLANNER_PROMPT.format(goal=self.task), images=[info["raw_frame"]])
            self._subgoals = parse_subgoals(response=out)
        obs, info = self.env.observe()
        self._before = (obs, info)
        prompt = action_prompt(goal=self.task, history=self._history(), ui_elements=obs["texts"]["ui_elements"],
                               guidance=hint, notes=self._notes, subgoals=self._subgoals, done=self._done,
                               context=device_context(env=self.env))
        response = self.call(tag="action", prompt=prompt, images=[info["raw_frame"], obs["frame"]])
        self._notes = read_notes(response=response) or self._notes
        if self._subgoals:
            self._done |= completed_subgoals(response=response, n=len(self._subgoals))
        reason, action = m3a_utils.parse_reason_action_output(response)
        if (not reason) or (not action):
            self._summaries.append(MSG_BAD_FORMAT)
            return Decision(invalid="bad format", error=MSG_BAD_FORMAT)
        parsed = agent_utils.extract_json(action)
        try:
            json_action.JSONAction(**parsed)        # M3A's validity test
        except Exception:
            self._summaries.append(MSG_BAD_ACTION)
            return Decision(invalid="unparseable action", error=MSG_BAD_ACTION)
        if self.task is None and parsed.get("action_type") in FINISH_ACTIONS:
            self._summaries.append(MSG_NO_FINISH)
            return Decision(invalid="finish while exploring", error=MSG_NO_FINISH)
        if parsed.get("action_type") == "status":
            return Decision(finish=True, answer=None, action_text=response)
        self._pending = {"action": action, "reason": reason,
                         "answer": parsed.get("text") if parsed.get("action_type") == "answer" else None}
        return Decision(action_text=response)

    def after_step(self, *, decision, obs_before, info_before, obs_after, info_after, step) -> None:
        pending = self._pending
        self._pending = None
        if pending and pending["answer"]:
            self.report.answer = pending["answer"]
        if not step.valid:
            # As M3A: the error becomes the summary, except a failed execution, which is not recorded.
            if step.error != MSG_EXEC_FAILED:
                self._summaries.append(step.error)
            return
        if self._before is not None:
            obs_before = self._before[0]     # the observation the action was chosen on
        # M3A.step effectively sends an unlabelled "before" and a labelled "after".
        before = np.array(obs_before["frame"], copy=True)
        after = np.array(obs_after["frame"], copy=True)
        m3a_utils.add_screenshot_label(after, "after")
        prompt = m3a._summarize_prompt(pending["action"], pending["reason"], self.task or EXPLORE_GOAL,
                                       obs_before["texts"]["ui_elements"], obs_after["texts"]["ui_elements"])
        current = self._current_call
        summary = self.call(tag="summary", prompt=prompt, images=[before, after])
        self._current_call = current   # the step stays owned by the action call
        self._summaries.append(f"Action selected: {pending['action']}. {summary}")
        note = readback_note(action=pending["action"], ui_elements=obs_after["texts"]["ui_elements"])   # F56
        if note:
            self._summaries[-1] += " " + note
