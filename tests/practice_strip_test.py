"""
Checks that stripping guidance from the final M3A and WebVoyager prompts gives the same prompt without guidance.
    python tests/practice_strip_test.py
"""
from cusi.agents.executors.base import strip_hint_blocks
from cusi.agents.executors.m3a import action_prompt
from cusi.agents.executors.webvoyager import init_message

guidance = "Summary: open contacts.\nSteps:\n  1. tap the + button\n  2. type the name"
ui = 'UI element 0: {"index": 0, "text": "Contacts", "is_clickable": True}\n'
history = ["Step 1- Action selected: {\"action_type\": \"open_app\"}. Opened."]
state = dict(history=history, ui_elements=ui, notes="contacts app is open", subgoals=["Open contacts"], done=set(),
             context="Device context: foreground app com.android.contacts; device date and time Monday.")
with_g = action_prompt(goal="add a contact", guidance=guidance, **state)
bare = action_prompt(goal="add a contact", guidance=None, **state)
assert guidance in with_g and "For The Current Task" in with_g
assert strip_hint_blocks(with_g) == bare, "M3A: stripped prompt differs from the prompt without guidance"

with_g = init_message(task="find papers", url="https://arxiv.org/", guidance=guidance)
bare = init_message(task="find papers", url="https://arxiv.org/", guidance=None)
assert guidance in with_g
assert strip_hint_blocks(with_g) == bare, "WebVoyager: stripped message differs"
print(bare)
print("STRIP TEST OK")
