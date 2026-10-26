"""
WebVoyager's agent with the final Web design (decisions.md E4, E8.0): one chat per leg, built as WebVoyager's run.py
builds it, plus the iteration budget in each observation (F10), no Google action (F14=google) and 5 screenshots kept
in the chat (F08=5), all fixed behaviour. WebVoyager's system prompt comes from the submodule and is patched here;
a missing anchor raises.
"""
import os
import re
from typing import Optional
from cusi.utils.log_handling import log_warn
from cusi.envs.webvoyager import MSG_NO_GOOGLE, load_webvoyager, without_google
from cusi.agents.executors.base import GUIDANCE_END, GUIDANCE_START, SECRET_NOTE, Decision, Executor

MAX_ATTACHED_IMGS = 5     # F08=5 (WebVoyager/run.sh: 3)
_PLACEHOLDER = "__CUSI_IMAGE__"   # passed to format_msg, whose image part is then swapped for an index
_PATTERN = r"Thought:|Action:|Observation:"
# run.py's messages, verbatim.
MSG_FORMAT = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
MSG_EXEC_FAILED = ("The action you have chosen cannot be exected. Please double-check if you have selected the wrong"
                   " Numerical Label or Action or Action format. Then provide the revised Thought and Action.")
OBS_PROMPT = "Observation: please analyze the attached screenshot and give the Thought and Action. "
NEW_TASK = "That task is finished. Now given a new task: {task} "
LOW_BUDGET_LEFT = 3


def init_message(*, task: str, url: str, guidance: Optional[str]) -> str:
    """run_single_task's init_msg, with the guidance block after the task sentence."""
    msg = f"""Now given a task: {task}  Please interact with https://www.example.com and get the answer. \n"""
    msg = msg.replace("https://www.example.com", url)
    if guidance:
        msg += f"{GUIDANCE_START}Guidance for this task:\n{guidance}\n{SECRET_NOTE}\n{GUIDANCE_END}"
    return msg + OBS_PROMPT


def budget_note(*, iteration: int, max_steps: int) -> str:
    """F10: the iteration count, appended to each observation; a warning in the last LOW_BUDGET_LEFT iterations."""
    left = max_steps - iteration + 1
    note = f"\n[Budget] This is iteration {iteration} of {max_steps}; {left - 1} more after this one."
    if left <= LOW_BUDGET_LEFT:
        note += " You are running out of iterations: answer as soon as you have the information."
    return note


class WebVoyagerExecutor(Executor):
    """WebVoyager with the final Web design on a WebVoyagerPlayEnv."""

    name = "webvoyager"
    env_name = "web"
    MAX_NEW_TOKENS = 1000     # run.py call_model
    TEMPERATURE = 1.0         # run.sh --temperature
    STOP_ON_MODEL_ERROR = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._wv = load_webvoyager(project_root=self._parameters["project_root"])
        self._messages: list = []
        self._images: list = []
        #: (iteration, set-of-mark frame) of every observation sent.
        self.screenshots: list = []
        self._it = 0
        self._fail_obs = ""
        self._pdf_obs = ""
        self._warn_obs = ""
        self._new_task: Optional[str] = None

    def reset_memory(self) -> None:
        # F14=google: no Google action (it often lands on a CAPTCHA)
        self._messages = [{"role": "system", "content": without_google(prompt=self._wv.SYSTEM_PROMPT)}]
        self._new_task = None
        self._images = []
        self._n_previous = 0
        self.screenshots = []
        self._it = 0
        self._fail_obs = self._pdf_obs = self._warn_obs = ""

    def history(self) -> dict:
        """The whole chat after the system prompt, with its images."""
        return {"messages": [dict(m) for m in self._messages[1:]], "images": list(self._images)}

    def restore_history(self, history: dict) -> None:
        self._messages = [self._messages[0]] + [dict(m) for m in history["messages"]]
        self._images = list(history["images"])
        self._n_previous = len(history["messages"])

    def _observation_message(self, *, obs: dict, init_msg: str) -> dict:
        msg = self._wv.format_msg(self._it, init_msg, self._pdf_obs, self._warn_obs, _PLACEHOLDER,
                                  obs["elements"])
        parts = []
        for part in msg["content"]:
            if part["type"] == "image_url":
                self._images.append(obs["frame"])
                self.screenshots.append((self._it, obs["frame"]))
                parts.append({"type": "image", "image": len(self._images) - 1})
            else:
                parts.append(dict(part))
        first_text = next(p for p in parts if p["type"] == "text")
        if self._new_task:      # retask: the chat's task is in its first message, so the new one is announced
            first_text["text"] = NEW_TASK.format(task=self._new_task) + first_text["text"]
            self._new_task = None
        first_text["text"] += budget_note(iteration=self._it, max_steps=self.report.max_steps)
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
        if chosen_action.startswith("Google"):
            self._fail_obs = MSG_NO_GOOGLE
            return Decision(invalid="pruned action", error=MSG_NO_GOOGLE)
        action_key, parsed = self._wv.extract_information(chosen_action)
        if action_key == "answer":
            return Decision(finish=True, answer=parsed["content"], action_text=f"Action: {chosen_action}")
        if action_key is None:
            # run.py gives the generic failure message for an unknown action.
            self._fail_obs = MSG_EXEC_FAILED
            return Decision(invalid="unknown action", error=MSG_EXEC_FAILED)
        return Decision(action_text=f"Action: {chosen_action}")

    def after_step(self, *, decision, obs_before, info_before, obs_after, info_after, step) -> None:
        self._fail_obs = "" if step.valid else (step.error or MSG_EXEC_FAILED)
        # the no-change note goes with the env's warnings into the next observation (WebVoyager's feedback channel)
        self._warn_obs = " ".join(w for w in (info_after.get("warning"), self.change_note) if w)
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

    def retask(self, *, task: str) -> None:
        super().retask(task=task)
        self._new_task = task

    def messages_for_log(self, *, own_leg_only: bool = True) -> list:
        """The chat in run.py's interact_messages.json format; own_leg_only drops chat carried over from earlier legs."""
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
        """This leg's messages; valid because clip_message_and_obs rewrites messages but never drops one."""
        return self._messages[1 + self._n_previous:]
