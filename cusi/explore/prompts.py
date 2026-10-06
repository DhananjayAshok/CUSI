"""Policy prompts for exploration: each benchmark's native executor prompt (Part 1), so what
the policy learns is in the evaluation format. Free play has no goal, so the goal slot holds
EXPLORE_GOAL.

    GameBoy  GameBoyRL's STEP_PROMPT with the `actions` history (as the practice executor).
    Android  M3A's action-selection prompt (raw + set-of-mark screenshots, element list). M3A's
             history is its own step summaries; to keep one model call per step, the policy's
             history lines are "Action selected: <action>" without the summary.
    Web      WebVoyager's system prompt + its first-turn observation message (screenshot +
             element list). No chat history: each step is a fresh first turn, which keeps
             training sequences short (three screenshots per step otherwise).

reply_format (curiosity_plan §8 Q6): "native" keeps each benchmark's reply format (reasoning +
action, ~150 tokens); "action_only" appends ACTION_ONLY[env] to the prompt, asking for the action
alone, and caps max_new_tokens at MAX_NEW_TOKENS[env] (sized to the longest valid action; typed
text makes Android/Web actions the longest). Every env parser accepts a bare action.

A PolicyPrompter keeps the per-episode memory and returns neutral chats (cusi.practice format).
"""
import json
from typing import Optional
from cusi.agents.records import EncodedImage, StepRecord

EXPLORE_GOAL = ("Explore: take actions that lead to screens and content you have not seen before in this"
                " session.")
HISTORY_K = 5
REPLY_FORMATS = ("action_only", "native")
ACTION_ONLY = {
    "gameboy": "Reply with only the action (for example `UP`), and nothing else: no reasoning.",
    "android": ('Reply with only the action JSON on one line (for example {"action_type": "click", "index": 3}), '
                "and nothing else: no reasoning."),
    "web": "Reply with only one line, `Action: <one action in the format above>`, and nothing else: no Thought.",
}
# Prefilled start of the reply in action_only mode. Qwen3.5-0.8B follows the native prompts' formats
# over the appended instruction: on Web it wrote "Thought: ..." and never reached the action within the
# token cap (0% valid in the first Web smoke run); on Android it wrote M3A's "Reason: ..." (79-96
# tokens). So the reply starts at the action itself.
RESPONSE_PREFIX = {"action_only": {"web": "Action:", "android": '{"action_type": "'}, "native": {}}
MAX_NEW_TOKENS = {"action_only": {"gameboy": 16, "android": 96, "web": 64},
                  "native": {"gameboy": 256, "android": 256, "web": 256}}


def _append_text(*, messages: list, text: str) -> list:
    """Add a final instruction to the last user message's text."""
    last = messages[-1]
    for part in last["content"]:
        if part.get("type") == "text":
            part["text"] = part["text"] + "\n\n" + text
            return messages
    last["content"].append({"type": "text", "text": text})
    return messages


def _single_turn(*, text: str, n_images: int) -> list:
    """cusi.utils sends the text first, then the images."""
    return [{"role": "user", "content": [{"type": "text", "text": text}]
             + [{"type": "image", "image": i} for i in range(n_images)]}]


class PolicyPrompter:
    def __init__(self, *, env_name: str, env, parameters: dict, reply_format: str = "action_only") -> None:
        if reply_format not in REPLY_FORMATS:
            raise ValueError(f"reply_format must be one of {REPLY_FORMATS}, got {reply_format!r}")
        self.reply_format = reply_format
        self.response_prefix = RESPONSE_PREFIX[reply_format].get(env_name)
        self.env_name = env_name
        self.env = env
        self._parameters = parameters
        self.history: list = []
        if env_name == "web":
            from cusi.envs.webvoyager import load_webvoyager
            self._wv = load_webvoyager(project_root=parameters["project_root"])

    def reset(self) -> None:
        self.history = []

    def build(self, *, obs: dict, info: dict, error: Optional[str]) -> tuple:
        """(messages, images) for the current state."""
        messages, images = self._build_native(obs=obs, info=info, error=error)
        if self.reply_format == "action_only":
            messages = _append_text(messages=messages, text=ACTION_ONLY[self.env_name])
        return messages, images

    def _build_native(self, *, obs: dict, info: dict, error: Optional[str]) -> tuple:
        if self.env_name == "gameboy":
            from cusi.agents.executors.gameboy import build_step_prompt, render_history
            text = build_step_prompt(task=EXPLORE_GOAL, hint=None, steps_taken=len(self.history), error=error,
                                     action_strings=self.env.action_strings(),
                                     history=render_history(steps=self.history))
            return _single_turn(text=text, n_images=1), [obs["frame"]]
        if self.env_name == "android":
            from cusi.agents.executors.m3a import action_prompt
            history = [f"Step {i + 1}- Action selected: {a}" for i, a in enumerate(self.history[-HISTORY_K:])]
            text = action_prompt(goal=EXPLORE_GOAL, history=history, ui_elements=obs["texts"]["ui_elements"],
                                 guidance=None)
            return _single_turn(text=text, n_images=2), [info["raw_frame"], obs["frame"]]
        from cusi.agents.executors.webvoyager import init_message
        init_msg = init_message(task=EXPLORE_GOAL, url=self.env.start_url, guidance=None)
        if error:
            init_msg = f"The previous action failed: {error}\n" + init_msg
        msg = self._wv.format_msg(1, init_msg, "", info.get("warning", "") or "", "__IMG__",
                                  obs["texts"]["web_elements"])
        parts = [{"type": "image", "image": 0} if p["type"] == "image_url" else dict(p) for p in msg["content"]]
        return [{"role": "system", "content": self._wv.SYSTEM_PROMPT}, {"role": "user", "content": parts}], \
            [obs["frame"]]

    def observe(self, *, generated: str, obs_before: dict, info_before: dict, obs_after: dict,
                info_after: dict) -> None:
        """Update the episode memory after a step."""
        if self.env_name == "gameboy":
            if info_after.get("parsed_action") is None:
                return      # GameBoyRL's history holds environment steps only
            frame = EncodedImage.of(obs_after["frame"])
            self.history.append(StepRecord(frame_before=frame, frame_after=frame, action_text=generated,
                                           parsed_action=info_after["parsed_action"], valid=info_after["valid"],
                                           error=info_after.get("error"),
                                           extra={k: info_after[k] for k in ("action_name", "frame_changed",
                                                                             "action_success", "low_level")
                                                  if k in info_after}))
        elif self.env_name == "android":
            from android_world.agents import m3a_utils
            _, action = m3a_utils.parse_reason_action_output(generated)
            if not action and info_after.get("parsed_action"):
                action = json.dumps(info_after["parsed_action"])      # action_only replies have no "Action:"
            self.history.append(action or "(unparseable output)")
