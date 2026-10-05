"""Check that cusi_practice's GameBoy executor reproduces GameBoyRL's prompts exactly.

Builds GameBoyRL's own PolicyExecutor (single action policy + actions history) without running
it, feeds its history policy the same steps, and compares PolicyExecutor._build_prompt and
the done-check prompt with ours, for: no hint / a hint, no error / an error, empty history /
a history with a [no change] step (which adds STUCK_HINT).

    python tests/practice_gameboy_prompt_test.py
"""
import types
import numpy as np
from cusi_utils import load_parameters
from cusi_envs.gameboy import GameBoyPlayEnv
from cusi_practice.executors import gameboy as ours
from cusi_practice.executors.base import strip_hint_blocks
from cusi_practice.records import EncodedImage, StepRecord
from execution.executors.executor import PolicyExecutor          # GameBoyRL (on sys.path via ours)
from execution.executors.policies.action import SingleActionPolicy
from execution.executors.policies.history import ActionHistoryPolicy
from execution.report import EnvironmentStepRecord, ExecutorReport
from gameboy_worlds.interface.action import LowLevelAction, LowLevelActions

parameters = load_parameters()
env = GameBoyPlayEnv(game="pokemon_red", init_state="location_viridian_city_starting_charmander",
                     parameters=parameters)
obs, info = env.reset()
frame = obs["frame"]
blank = EncodedImage.of(frame)

# The same three steps, recorded both ways: UP (moved), A (no change), LEFT (moved).
moves = [("UP", LowLevelActions.PRESS_ARROW_UP, True), ("A", LowLevelActions.PRESS_BUTTON_A, False),
         ("LEFT", LowLevelActions.PRESS_ARROW_LEFT, True)]
gb_steps = [EnvironmentStepRecord(frame_before=frame, frame_after=frame, action_class=LowLevelAction,
                                  kwargs={"low_level_action": lla}, transition_states=[], action_success=0,
                                  frame_changed=changed) for _, lla, changed in moves]
our_steps = [StepRecord(frame_before=blank, frame_after=blank, action_text=name, parsed_action=None, valid=True,
                        error=None, extra={"action_name": name, "frame_changed": changed, "low_level": True,
                                           "action_success": 0}) for name, _, changed in moves]


def gameboyrl_prompt(*, task, hint, error, n_history):
    ex = PolicyExecutor.__new__(PolicyExecutor)
    ex._task, ex._hint, ex._env = task, hint, env._env
    ex._action_policy = SingleActionPolicy()
    ex._history_policy = ActionHistoryPolicy(history_k=5)
    ex._history_policy.observe(gb_steps[:n_history])
    report = ExecutorReport.__new__(ExecutorReport)
    report.vlm_call_log = []
    ex.report = types.SimpleNamespace(steps=gb_steps[:n_history])
    return ex._build_prompt(error)


def our_prompt(*, task, hint, error, n_history):
    return ours.build_step_prompt(task=task, hint=hint, steps_taken=n_history, error=error,
                                  action_strings=env.action_strings(),
                                  history=ours.render_history(steps=our_steps[:n_history]))


n_checked = 0
for hint in (None, "Walk up to the Pokemon Center door."):
    for error in (None, "Your previous response could not be parsed."):
        for n_history in (0, 1, 3):
            theirs = gameboyrl_prompt(task="enter the pokemon center", hint=hint, error=error, n_history=n_history)
            mine = our_prompt(task="enter the pokemon center", hint=hint, error=error, n_history=n_history)
            assert theirs == mine, f"prompt mismatch (hint={hint!r}, error={error!r}, n={n_history}):\n" \
                                   f"--- GameBoyRL\n{theirs}\n--- ours\n{mine}"
            # Stripping the hint gives the no-hint prompt (what create_dataset trains on).
            if hint is not None:
                bare = our_prompt(task="enter the pokemon center", hint=None, error=error, n_history=n_history)
                assert strip_hint_blocks(mine) == bare, "stripped prompt differs from the no-hint prompt"
            n_checked += 1
assert "[no change]" in our_prompt(task="t", hint=None, error=None, n_history=3)
print(f"GameBoy executor prompts match GameBoyRL's ({n_checked} combinations).")
print("--- example prompt (hint, history 3) ---")
print(our_prompt(task="enter the pokemon center", hint="Walk up.", error=None, n_history=3))
env.close()
print("PROMPT TEST OK")
