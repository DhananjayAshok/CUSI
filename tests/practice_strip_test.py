"""Check that stripping the guidance block from M3A's and WebVoyager's prompts gives exactly the
prompt each native agent builds without guidance (what create_dataset trains on).

    python tests/practice_strip_test.py
"""
from cusi_practice.executors.base import strip_hint_blocks
from cusi_practice.executors.m3a import action_prompt
from cusi_practice.executors.webvoyager import init_message
from android_world.agents import m3a

guidance = "Summary: open contacts.\nSteps:\n  1. tap the + button\n  2. type the name"
ui = 'UI element 0: {"index": 0, "text": "Contacts", "is_clickable": True}\n'
history = ["Step 1- Action selected: {\"action_type\": \"open_app\"}. Opened."]
with_g = action_prompt(goal="add a contact", history=history, ui_elements=ui, guidance=guidance)
native = m3a._action_selection_prompt("add a contact", history, ui, None)
assert guidance in with_g and "For The Current Task" in with_g
assert strip_hint_blocks(with_g) == native, "M3A: stripped prompt differs from the native prompt"
assert action_prompt(goal="add a contact", history=history, ui_elements=ui, guidance=None) == native

with_g = init_message(task="find papers", url="https://arxiv.org/", guidance=guidance)
bare = init_message(task="find papers", url="https://arxiv.org/", guidance=None)
assert guidance in with_g
assert strip_hint_blocks(with_g) == bare, "WebVoyager: stripped message differs"
print(bare)
print("STRIP TEST OK")
