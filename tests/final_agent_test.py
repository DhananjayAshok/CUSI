"""
Checks the final agents' prompts (decisions.md E8) on the real WebVoyager, M3A and GameBoyRL prompt text, with and
without a task, and the GameBoy subgoal supervisor's leg rules. No model calls.
    python tests/final_agent_test.py              # everything (the GameBoy check runs in its own process)
"""
import collections
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                                                   # noqa: E402
from cusi.agents.executors.base import EXPLORE_GOAL                                  # noqa: E402
from cusi.agents.records import EncodedImage, StepRecord                            # noqa: E402

UI = ('UI element 0: {"index": 0, "resource_id": "title", "text": "Notes", "is_editable": False}\n'
      'UI element 1: {"index": 1, "text": "groceries", "content_description": "", "is_editable": True}\n')


def check_web() -> None:
    from cusi.envs.webvoyager import load_webvoyager
    from cusi.agents.executors import webvoyager as wv
    base = load_webvoyager(project_root=ROOT).SYSTEM_PROMPT
    task_prompt = wv.system_prompt(base=base, explore=False)
    removed = [line for line in base.splitlines() if line not in task_prompt.splitlines()]
    assert len(removed) == 2 and all("Google" in line for line in removed), removed
    explore_prompt = wv.system_prompt(base=base, explore=True)
    removed = [line for line in task_prompt.splitlines() if line not in explore_prompt.splitlines()]
    assert len(removed) == 2 and all("Answer" in line or "ANSWER" in line for line in removed), removed
    assert wv.budget_note(iteration=1, max_steps=15).endswith("iteration 1 of 15; 14 more after this one.")
    assert "running out" not in wv.budget_note(iteration=12, max_steps=15)
    assert "running out" in wv.budget_note(iteration=13, max_steps=15)
    assert wv.MAX_ATTACHED_IMGS == 5
    from cusi.envs.webvoyager import action_menu
    for mode in ("test", "free_play"):           # search and practice read the env's menu, not the prompt
        menu = action_menu(system_prompt=base, mode=mode)
        assert "Google" not in menu and "Click [Numerical_Label]" in menu, menu
        assert ("ANSWER" in menu) == (mode == "test"), mode
    print("web: Google removed from prompt and env menu, ANSWER removed when exploring, budget note, 5 screenshots")


def check_android() -> None:
    from cusi.agents.executors import m3a as ex
    from cusi.envs import android_world as aw
    for mode in ("test", "free_play"):           # search and practice read the env's menu, not the prompt
        menu = aw.action_menu(mode=mode)
        assert aw.CLEAR_TEXT_NEW.split("`")[0] in menu and aw.INPUT_DESC_NEW in menu and "pressing the enter" not in menu
        assert ('"action_type": "status"' in menu) == (mode == "test"), mode
    kw = dict(history=["Step 1- opened Markor"], ui_elements=UI, guidance=None, notes="list has 3 items",
              subgoals=["Open the note", "Add the item"], done={0}, context="Device context: foreground app x.")
    prompt = ex.action_prompt(goal="Add milk to the groceries note", **kw)
    marks = [ex.NOTES_SHOWN + "list has 3 items", "Additional rules:", ex.SUBGOALS_HEADER, "Device context:",
             "Your Answer:"]
    positions = [prompt.rfind(m) for m in marks]
    assert all(p >= 0 for p in positions) and positions == sorted(positions), positions
    assert "1. [x] Open the note" in prompt and "2. [ ] Add the item" in prompt
    assert aw.CLEAR_TEXT_NEW in prompt and aw.INPUT_DESC_NEW in prompt and ex.DELETE_TIP_NEW in prompt
    assert "remember to delete them before typing" not in prompt
    assert '"action_type": "status"' in prompt and "Add milk" in prompt

    explore = ex.action_prompt(goal=None, **{**kw, "subgoals": [], "done": set()})
    assert EXPLORE_GOAL in explore and ex.SUBGOALS_HEADER not in explore
    assert not any(line.startswith("- ") and ('"action_type": "status"' in line or '"action_type": "answer"' in line)
                   for line in explore.split("\n"))

    assert ex.read_notes(response='Reason: r\nNotes: milk is\n  missing\nAction: {"action_type": "wait"}') == \
        "milk is missing"
    assert ex.parse_subgoals(response='```json\n{"subgoals": ["a", "b"]}\n```') == ["a", "b"]
    assert ex.parse_subgoals(response="no plan") == []
    assert ex.completed_subgoals(response="Completed subgoals: 1, 3\nAction: x", n=2) == {0}
    typed = '{"action_type": "input_text", "text": "groceries", "index": 1}'
    assert "contains" in ex.readback_note(action=typed, ui_elements=UI)
    assert "is not in any text field" in ex.readback_note(action=typed.replace("groceries", "milk"), ui_elements=UI)
    assert ex.readback_note(action='{"action_type": "click", "index": 0}', ui_elements=UI) is None
    print("android: env menu = prompt actions, blocks in the confirmed order, F55/F57 edits, no-task prompt without finishing or plan")


def check_gameboy() -> None:
    from cusi.agents.executors import gameboy as gb
    prompt = gb.build_step_prompt(task=None, hint=None, steps_taken=0, error=None, action_strings={"a": "UP"},
                                  history="")
    assert EXPLORE_GOAL in prompt and "[TASK]" not in prompt
    from cusi.agents.agent import _classes
    assert tuple(c.__name__ for c in _classes("gameboy")) == ("GameBoyExecutor", "SubgoalSupervisor")
    print("gameboy: no-task prompt carries the explore goal; the executor runs under the subgoal supervisor")


def step(before: int, after: int) -> StepRecord:
    frame = lambda v: EncodedImage.of(np.full((4, 4, 3), v, dtype=np.uint8))
    return StepRecord(frame_before=frame(before), frame_after=frame(after), action_text="UP", parsed_action=None,
                      valid=True, error=None, reward=0.0, extra={})


def check_subgoal() -> None:
    import types
    from cusi.agents.records import CallRecord, LegReport
    from cusi.agents.supervisors import subgoal as sg
    spec = lambda **kw: sg.SubgoalSupervisor._leg_spec(types.SimpleNamespace(_task="T"), **kw)
    assert spec(target="s1", is_last=False, hint=None) == ("s1", None, True)          # a plan step self-terminates
    assert spec(target="s3", is_last=True, hint=None) == ("T", "s3", False)           # the last step runs the task
    assert spec(target="s3", is_last=True, hint="h") == ("T", "h", False)
    assert spec(target=None, is_last=True, hint=None) == ("T", None, False)           # no plan
    block = sg.plan_block(targets=["a", "b", "c"], index=1)
    assert block == "  1. a  [DONE]\n  2. b  <-- CURRENT, just failed\n  3. c  [not yet attempted]", block
    act = CallRecord(tag="action", images=[], response="Reasoning: go up\nAction: UP", prompt="p")
    act.steps.append(step(1, 2))
    done = CallRecord(tag="done_check", images=[], response="Reasoning: not yet\nComplete: no", prompt="p")
    report = LegReport(env_name="gameboy", task="s1", hint=None, max_steps=10, calls=[act, done])
    trace = sg.action_trace(report)
    assert trace == "         1. UP — go up\n            ↳ finished? no — not yet", trace
    assert sg.budget_spent(report) == 1
    print("subgoal: plan steps self-terminate, last step runs the task, plan marking, trace with done-checks")


def check_factory() -> None:
    from cusi.agents.agent import _classes
    names = {env: tuple(c.__name__ for c in _classes(env)) for env in ("web", "android")}
    assert names == {"web": ("WebVoyagerExecutor", "DirectSupervisor"),
                     "android": ("M3AExecutor", "DirectSupervisor")}, names
    print("factory: web and android run their executor with no supervisor")


if __name__ == "__main__":
    if "--gameboy" in sys.argv:      # GameBoyRL's and WebVoyager's `utils` modules cannot share a process
        check_gameboy()
        sys.exit(0)
    check_web()
    check_android()
    check_subgoal()
    check_factory()
    if "--no_gameboy" not in sys.argv:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--gameboy"], check=True)
    print("FINAL AGENT TEST OK")
