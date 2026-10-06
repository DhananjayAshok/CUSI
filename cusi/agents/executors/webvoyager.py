"""WebVoyager's agent inside the shared skeleton.

One chat for the whole leg, built exactly as WebVoyager/run.py's run_single_task builds it:
SYSTEM_PROMPT, the task message, format_msg for each observation (set-of-mark screenshot +
the web element list), the failure message as a plain user turn after a failed action,
clip_message_and_obs keeping the last 3 screenshots (run.sh's --max_attached_imgs), and
run.py's format check ("Thought:" and "Action:" both required). ANSWER ends the leg with the
answer kept for the judge (cusi.eval sends it through env.step, so the test-mode env terminates).
`messages_for_log()` gives the chat as run.py's print_message writes interact_messages.json.

Images live in the neutral chat as {"type": "image", "image": i} parts: format_msg is called
with a placeholder base64 string, and its image part is swapped for an index.

Guidance is a marked block in the first user message, after the task sentence.
"""
import os
import re
from typing import Optional
from cusi.utils.log_handling import log_warn
from cusi.envs.webvoyager import load_webvoyager
from cusi.agents.executors.base import GUIDANCE_END, GUIDANCE_START, SECRET_NOTE, Decision, Executor

MAX_ATTACHED_IMGS = 3     # WebVoyager/run.sh --max_attached_imgs
_PLACEHOLDER = "__CUSI_IMAGE__"
_PATTERN = r"Thought:|Action:|Observation:"
# run.py's messages, verbatim.
MSG_FORMAT = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
MSG_EXEC_FAILED = ("The action you have chosen cannot be exected. Please double-check if you have selected the wrong"
                   " Numerical Label or Action or Action format. Then provide the revised Thought and Action.")
OBS_PROMPT = "Observation: please analyze the attached screenshot and give the Thought and Action. "


def init_message(*, task: str, url: str, guidance: Optional[str]) -> str:
    """run_single_task's init_msg, with the guidance block after the task sentence."""
    msg = f"""Now given a task: {task}  Please interact with https://www.example.com and get the answer. \n"""
    msg = msg.replace("https://www.example.com", url)
    if guidance:
        msg += f"{GUIDANCE_START}Guidance for this task:\n{guidance}\n{SECRET_NOTE}\n{GUIDANCE_END}"
    return msg + OBS_PROMPT


class WebVoyagerExecutor(Executor):
    """WebVoyager on a WebVoyagerPlayEnv."""

    name = "webvoyager"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._wv = load_webvoyager(project_root=self._parameters["project_root"])
        self._messages: list = []
        self._images: list = []
        #: (iteration, set-of-mark frame) of every observation sent, as run.py saves
        #: screenshot{it}.png (read by cusi.eval.web to write the native artefacts).
        self.screenshots: list = []
        self._it = 0
        self._fail_obs = ""
        self._pdf_obs = ""
        self._warn_obs = ""

    def reset_memory(self) -> None:
        self._messages = [{"role": "system", "content": self._wv.SYSTEM_PROMPT}]
        self._images = []
        self._n_previous = 0
        self.screenshots = []
        self._it = 0
        self._fail_obs = self._pdf_obs = self._warn_obs = ""

    def history(self) -> dict:
        """WebVoyager's memory: the whole chat after the system prompt, with its images. The next
        leg continues it with its own first message (task, hint and current screenshot)."""
        return {"messages": [dict(m) for m in self._messages[1:]], "images": list(self._images)}

    def restore_history(self, history: dict) -> None:
        self._messages = [self._messages[0]] + [dict(m) for m in history["messages"]]
        self._images = list(history["images"])
        self._n_previous = len(history["messages"])

    def _observation_message(self, *, obs: dict, init_msg: str) -> dict:
        msg = self._wv.format_msg(self._it, init_msg, self._pdf_obs, self._warn_obs, _PLACEHOLDER,
                                  obs["texts"]["web_elements"])
        parts = []
        for part in msg["content"]:
            if part["type"] == "image_url":
                self._images.append(obs["frame"])
                self.screenshots.append((self._it, obs["frame"]))
                parts.append({"type": "image", "image": len(self._images) - 1})
            else:
                parts.append(dict(part))
        return {"role": "user", "content": parts}

    def _compact(self, messages: list) -> tuple:
        """Only the images the (clipped) chat still references, re-indexed."""
        used, images, out = {}, [], []
        for msg in messages:
            if isinstance(msg["content"], str):
                out.append(dict(msg))
                continue
            parts = []
            for p in msg["content"]:
                if p["type"] == "image":
                    if p["image"] not in used:
                        used[p["image"]] = len(images)
                        images.append(self._images[p["image"]])
                    parts.append({"type": "image", "image": used[p["image"]]})
                else:
                    parts.append(dict(p))
            out.append({"role": msg["role"], "content": parts})
        return out, images

    def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
        self._it += 1
        if not self._fail_obs:
            init_msg = init_message(task=self.task, url=self.env.start_url, guidance=hint)
            self._messages.append(self._observation_message(obs=obs, init_msg=init_msg))
        else:
            self._messages.append({"role": "user", "content": self._fail_obs})
        self._messages = self._wv.clip_message_and_obs(self._messages, MAX_ATTACHED_IMGS)
        messages, images = self._compact(self._messages)
        response = self.chat_call(tag="action", messages=messages, images=images)
        self._messages.append({"role": "assistant", "content": response})
        self._fail_obs = ""
        if "Thought:" not in response or "Action:" not in response:
            self._fail_obs = MSG_FORMAT
            return Decision(invalid="format error", error=MSG_FORMAT)
        chosen_action = re.split(_PATTERN, response)[2].strip()
        action_key, parsed = self._wv.extract_information(chosen_action)
        if action_key == "answer":
            # action_text: what env.step parses if the caller sends the finish through the env.
            return Decision(finish=True, answer=parsed["content"], action_text=f"Action: {chosen_action}")
        if action_key is None:
            # run.py: exec_action raises NotImplementedError -> the generic failure message.
            self._fail_obs = MSG_EXEC_FAILED
            return Decision(invalid="unknown action", error=MSG_EXEC_FAILED)
        return Decision(action_text=f"Action: {chosen_action}")

    def after_step(self, *, decision, obs_before, info_before, obs_after, info_after, step) -> None:
        self._fail_obs = "" if step.valid else (step.error or MSG_EXEC_FAILED)
        self._warn_obs = info_after.get("warning", "") or ""
        self._pdf_obs = self._pdf_answer(info_after) if info_after.get("pdf") else ""

    def _pdf_answer(self, info: dict) -> str:
        """run.py's PDF observation: the model answers the task from the PDF's text."""
        from benchmark_adapters.webvoyager_adapter import answer_from_pdf
        name = info["pdf"].split("You downloaded a PDF file: ", 1)[-1].rstrip(".")
        path = os.path.join(self.env._work_dir, name)
        try:
            answer = answer_from_pdf(model=self.vlm._model, pdf_path=path, question=self.task,
                                     parameters=self._parameters)
        except Exception as e:
            log_warn(f"PDF answer failed for {path}: {e}", parameters=self._parameters)
            return info["pdf"]
        return ("You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get"
                " the following response: " + answer)

    def messages_for_log(self, *, own_leg_only: bool = True) -> list:
        """The (clipped) chat in run.py's interact_messages.json format: OpenAI-style parts, every
        image replaced by run.py's placeholder URL. own_leg_only: the system prompt and this leg's
        own messages (what auto_eval reads: its first user message holds the real task), without
        the chat carried over from earlier legs."""
        out = []
        messages = self._messages
        if own_leg_only and self._n_previous:
            messages = [self._messages[0]] + self._own_messages()
        for msg in messages:
            if isinstance(msg["content"], str):
                out.append({"role": msg["role"], "content": msg["content"]})
                continue
            out.append({"role": msg["role"], "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,{b64_img}"}}
                if p["type"] == "image" else {"type": "text", "text": p["text"]} for p in msg["content"]]})
        return out

    def _own_messages(self) -> list:
        """This leg's messages: those after the carried-over chat. clip_message_and_obs rewrites
        messages in place (image parts become text) but never drops one, so the count holds."""
        return self._messages[1 + self._n_previous:]
