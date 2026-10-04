"""WebVoyager as a TextActionEnv (see cusi_envs/base.py).

    env = WebVoyagerPlayEnv(task_id="Allrecipes--0", mode="test")
    env = WebVoyagerPlayEnv(url="https://www.allrecipes.com/", mode="free_play")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step("Thought: search\\nAction: Type [3]; lasagna")

The scene is a start URL (a WebVoyager task's `web`, or any URL), fixed at
construction. reset() quits the browser and opens a fresh one, with a fresh profile,
on that URL. The sites are live, so a reset restores the browser, not the page
content: sites change, rate-limit, and may show cookie banners or CAPTCHAs.

Everything that touches the browser is WebVoyager's own code from run.py/utils.py:
driver_config + _make_driver (browser), open_start_page (page open), get_web_element_rect
(set-of-mark labels + element text), extract_information (action parsing) and
exec_action (action execution, with eval's failure/warning feedback).

Observation: frame = the set-of-mark screenshot (HxWx3 uint8); texts =
{"web_elements": WebVoyager's labelled element list}; actions = the action list from
WebVoyager's system prompt; goal = the task question in "test" mode, "" otherwise.

Modes (base.py):
    test       ANSWER ends the episode, with the answer in info["answer"]; reward 0.
               WebVoyager's success signal is its LLM judge (evaluation/auto_eval.py),
               applied afterwards to the answer and final screenshots, as evaluation
               does now. Truncates at max_steps (default 15, run.sh's --max_iter).
    free_play  no goal, ANSWER rejected, reward 0, no step limit.

As in evaluation, an action that cannot be executed returns the previous observation
(same labels, info["stale"] = True) with WebVoyager's failure message; it does not
re-observe the page.

fast_waits=True replaces run.py's fixed post-action sleeps (2-10 s) with "wait until the
page has loaded", capped at the original duration (sets run.FAST_WAITS for the process).

Needs the CUSI container (Chromium + chromedriver): run via scripts/container.sh. No KVM.
"""
import argparse
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
from typing import Any, Optional
import numpy as np
from PIL import Image
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_info, log_warn, log_error
from cusi_envs.base import TextActionEnv, scene_hash

TEXT_KEYS = ("web_elements",)
DEFAULT_TEST_MAX_STEPS = 15   # WebVoyager/run.sh --max_iter

# run.py's format error, verbatim.
MSG_FORMAT = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
MSG_NO_ANSWER = "The ANSWER action is not available: there is no task to answer. No action was performed."


def load_webvoyager(*, project_root: str):
    """Import WebVoyager's run.py (a script, which imports its siblings `utils` and
    `prompts` by bare name) as the module `webvoyager_run`."""
    if "webvoyager_run" in sys.modules:
        return sys.modules["webvoyager_run"]
    wv_dir = os.path.join(project_root, "WebVoyager")
    if wv_dir not in sys.path:
        sys.path.insert(0, wv_dir)
    spec = importlib.util.spec_from_file_location("webvoyager_run", os.path.join(wv_dir, "run.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["webvoyager_run"] = module
    spec.loader.exec_module(module)
    return module


def action_menu(*, system_prompt: str, mode: str) -> str:
    """The action descriptions and formats from WebVoyager's system prompt. free_play
    drops ANSWER."""
    numbered = re.search(r"\n(1\. Click.*?)\n\n", system_prompt, flags=re.DOTALL).group(1)
    formats = re.search(r"STRICTLY follow the format:\n(.*?)\n\n", system_prompt, flags=re.DOTALL).group(1)
    lines = numbered.splitlines() + ["", "Action format:"] + formats.splitlines()
    if mode == "free_play":
        lines = [l for l in lines if not l.startswith("7. Answer") and "ANSWER" not in l]
    return "\n".join(l.rstrip() for l in lines)


def action_text(*, text: str) -> str:
    """The part extract_information should see: what follows the last 'Action:'."""
    return text.rsplit("Action:", 1)[1].strip() if "Action:" in text else text.strip()


def canonical_action(*, action_key: str, info: Any) -> dict:
    if action_key == "click":
        return {"action_type": "click", "index": int(info[0])}
    if action_key == "type":
        return {"action_type": "type", "index": int(info["number"]), "text": info["content"]}
    if action_key == "scroll":
        target = info["number"]
        return {"action_type": "scroll", "target": target if target == "WINDOW" else int(target),
                "direction": info["content"]}
    if action_key == "answer":
        return {"action_type": "answer", "text": info["content"]}
    return {"action_type": action_key}


def load_task(*, task_id: str, data_file: str) -> dict:
    with open(data_file) as f:
        for line in f:
            task = json.loads(line)
            if task["id"] == task_id:
                return task
    raise KeyError(f"No WebVoyager task {task_id!r} in {data_file}")


class WebVoyagerPlayEnv(TextActionEnv):
    """A WebVoyager start page as a text-action gym.Env. See the module docstring."""

    def __init__(
        self,
        *,
        task_id: Optional[str] = None,
        url: Optional[str] = None,
        mode: str = "test",
        data_file: Optional[str] = None,
        max_steps: Optional[int] = None,
        fast_waits: bool = False,
        headless: bool = True,
        window_width: int = 1024,
        window_height: int = 768,
        fix_box_color: bool = True,
        chrome_binary: Optional[str] = None,
        chromedriver: Optional[str] = None,
        work_dir: Optional[str] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        :param task_id: A WebVoyager task id, e.g. "Allrecipes--0" (needed for test mode).
        :param url: free_play only: a start URL instead of a task.
        :param data_file: Task file. Default WebVoyager/data/WebVoyager_data.jsonl.
        :param max_steps: Truncate after this many steps. None: 15 in test mode, unlimited
            in free_play.
        :param fast_waits: See the module docstring.
        :param chrome_binary, chromedriver: Default: the container's ($CUSI_CHROME_BINARY,
            $CUSI_CHROMEDRIVER).
        :param work_dir: Browser profile + downloads. Default: a new /tmp directory.
        """
        self._parameters = load_parameters(parameters)
        self._wv = load_webvoyager(project_root=self._parameters["project_root"])
        self._wv.FAST_WAITS = fast_waits
        data_file = data_file or os.path.join(self._parameters["project_root"], "WebVoyager", "data",
                                              "WebVoyager_data.jsonl")
        if task_id is not None:
            self._task = load_task(task_id=task_id, data_file=data_file)
        elif url is not None and mode == "free_play":
            self._task = {"id": None, "web_name": None, "ques": "", "web": url}
        else:
            log_error("Pass task_id (or, in free_play, url)", parameters=self._parameters)
        self.start_url = self._task["web"]
        self.scene_id = scene_hash(parts=("webvoyager", self._task["id"], self.start_url))
        self._work_dir = work_dir or tempfile.mkdtemp(prefix=f"{os.environ.get('USER', 'user')}-cusi-wv-")
        self._download_dir = os.path.join(self._work_dir, "downloads")
        self._args = argparse.Namespace(
            headless=headless, window_width=window_width, window_height=window_height,
            fix_box_color=fix_box_color, text_only=False, save_accessibility_tree=False,
            force_device_scale=False, download_dir=self._download_dir,
            chrome_binary=chrome_binary or os.environ.get("CUSI_CHROME_BINARY"),
            chromedriver=chromedriver or os.environ.get("CUSI_CHROMEDRIVER"),
            chromedriver_log=None, chrome_profile_dir=None, chrome_disable_shm=False)
        self._driver = None
        self._web_eles = []
        self._last_obs = None
        self._url = self.start_url
        self.browser_restarts = 0

        frame = self._open(url=self.start_url)
        super().__init__(mode=mode, frame_shape=frame.shape, text_keys=TEXT_KEYS, parameters=self._parameters)
        if max_steps is None and mode == "test":
            max_steps = DEFAULT_TEST_MAX_STEPS
        self.max_steps = max_steps
        self.actions_text = action_menu(system_prompt=self._wv.SYSTEM_PROMPT, mode=mode)
        self._goal = self._task["ques"] if mode == "test" else ""
        self._steps = 0
        self._at_initial_state = True
        log_info(f"WebVoyagerPlayEnv ready: {self._task['id'] or self.start_url} (mode {mode}, "
                 f"fast_waits {fast_waits})", parameters=self._parameters)

    # ---------------------------------------------------------------- browser

    def _new_browser(self) -> None:
        self._quit()
        profile = os.path.join(self._work_dir, "profile")
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(self._download_dir, ignore_errors=True)
        os.makedirs(self._download_dir)
        self._args.chrome_profile_dir = profile
        options = self._wv.driver_config(self._args)
        self._driver = self._wv._make_driver(self._args, options)
        self._download_files = []

    def _quit(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def _open(self, *, url: str) -> np.ndarray:
        """Fresh browser on `url`; returns the first frame (and sets the observation)."""
        self._new_browser()
        self._wv.open_start_page(self._driver, url)
        return self._observe()[0]

    def _observe(self):
        """Set-of-mark screenshot + labelled element text, as run.py builds them."""
        rects, web_eles, web_eles_text = self._wv.get_web_element_rect(self._driver,
                                                                        fix_color=self._args.fix_box_color)
        png = self._driver.get_screenshot_as_png()
        try:
            for rect in rects:
                self._driver.execute_script("arguments[0].remove()", rect)
        except self._wv.StaleElementReferenceException:
            pass
        frame = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)
        self._web_eles = web_eles
        try:
            self._url = self._driver.current_url
        except Exception:
            pass
        self._last_obs = (frame, web_eles_text)
        return self._last_obs

    def _obs(self) -> dict:
        frame, web_eles_text = self._last_obs
        return {"frame": frame, "texts": {"web_elements": web_eles_text},
                "actions": self.actions_text, "goal": self._goal}

    def _on_pdf(self, pdf_path: str) -> str:
        shutil.copy(pdf_path, self._work_dir)
        return f"You downloaded a PDF file: {os.path.basename(pdf_path)}."

    # ---------------------------------------------------------------- gym API

    def _reset_impl(self):
        if not self._at_initial_state:
            self._open(url=self.start_url)
        self._at_initial_state = False
        self._steps = 0
        info = self.make_info(valid=True, error=None, parsed_action=None, step=0, scene_id=self.scene_id,
                              url=self._url)
        return self._obs(), info

    def _step_impl(self, action: str):
        self._at_initial_state = False
        self._steps += 1
        action_key, parsed_info = self._wv.extract_information(action_text(text=action))
        truncated = self.max_steps is not None and self._steps >= self.max_steps

        if action_key is None:
            info = self.make_info(valid=False, error=MSG_FORMAT, parsed_action=None, step=self._steps,
                                  scene_id=self.scene_id, stale=True)
            return self._obs(), 0.0, False, truncated, info
        parsed = canonical_action(action_key=action_key, info=parsed_info)

        if action_key == "answer":
            if self.mode == "test":
                info = self.make_info(valid=True, error=None, parsed_action=parsed, step=self._steps,
                                      scene_id=self.scene_id, answer=parsed["text"], url=self._url)
                return self._obs(), 0.0, True, False, info
            info = self.make_info(valid=False, error=MSG_NO_ANSWER, parsed_action=parsed, step=self._steps,
                                  scene_id=self.scene_id, stale=True)
            return self._obs(), 0.0, False, truncated, info

        extra = {}
        try:
            fail_obs, pdf_obs, warn_obs, self._download_files = self._wv.exec_action(
                action_key, parsed_info, self._driver, self._web_eles, None, self._args,
                self.start_url, self._download_files, self._on_pdf)
            if pdf_obs:
                extra["pdf"] = pdf_obs
            if warn_obs:
                extra["warning"] = warn_obs
            if fail_obs:
                # As in evaluation: no re-observation; the agent retries on the same labels.
                info = self.make_info(valid=False, error=fail_obs, parsed_action=parsed, step=self._steps,
                                      scene_id=self.scene_id, stale=True, **extra)
                return self._obs(), 0.0, False, truncated, info
            self._observe()
        except self._wv.WebDriverException as e:
            if not self._wv._is_browser_dead(e):
                raise
            if self.mode == "test":
                # Evaluation retries a crashed task from scratch: end this episode.
                info = self.make_info(valid=False, error="The browser crashed.", parsed_action=parsed,
                                      step=self._steps, scene_id=self.scene_id, browser_died=str(e)[:200])
                return self._obs(), 0.0, False, True, info
            log_warn(f"Browser died ({str(e)[:120]}); restarting on {self._url}", parameters=self._parameters)
            self.browser_restarts += 1
            self._open(url=self._url)
            extra["browser_restarted"] = True
        info = self.make_info(valid=True, error=None, parsed_action=parsed, step=self._steps,
                              scene_id=self.scene_id, url=self._url, **extra)
        return self._obs(), 0.0, False, truncated, info

    def close(self) -> None:
        self._quit()
        shutil.rmtree(self._work_dir, ignore_errors=True)
