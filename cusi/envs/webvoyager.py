"""
WebVoyager as a TextActionEnv, driving the browser with WebVoyager's own run.py code.
Sites are live, so reset restores the browser, not the page content. Reward is 0: the LLM judge scores afterwards.
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
import threading
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info, log_warn, log_error
from cusi.envs.base import TextActionEnv, scene_hash

TEXT_KEYS = ("web_elements",)
_BROWSER_START_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
DEFAULT_TEST_MAX_STEPS = 15   # WebVoyager/run.sh --max_iter

# run.py's format error, verbatim.
MSG_FORMAT = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
MSG_NO_ANSWER = "The ANSWER action is not available: there is no task to answer. No action was performed."


# --------------------------------------------------------------------------- bot-check pages
# A challenge page (Cloudflare, captcha, bot block) is not site content; the search scores it 0 and never expands it.

CHALLENGE_TITLES = ("just a moment", "attention required", "access denied", "are you a robot", "are you a human",
                    "robot or human", "security check", "captcha", "human verification", "pardon our interruption",
                    "verify you are human", "bot verification", "request blocked")
CHALLENGE_PHRASES = ("verify you are human", "verifying you are human", "checking your browser",
                     "checking if the site connection is secure", "unusual traffic", "are you a robot",
                     "press & hold", "press and hold", "complete the security check", "you have been blocked",
                     "enable javascript and cookies to continue", "pardon our interruption", "request unsuccessful",
                     "access denied", "not a robot", "solve the captcha", "complete the captcha")
#: A matched phrase or marker counts only on a page with less body text than this (a real page may mention them).
CHALLENGE_MAX_TEXT = 1500
# [title, first 3000 chars of the body text, whether a bot-check widget/iframe is in the DOM]
_CHALLENGE_JS = """
const sel = '#challenge-form, #challenge-running, #challenge-stage, #cf-challenge-running, .cf-turnstile,'
  + ' [name="cf-turnstile-response"], iframe[src*="challenges.cloudflare.com"], iframe[src*="hcaptcha.com"],'
  + ' iframe[src*="recaptcha"], .h-captcha, .g-recaptcha, #px-captcha, #captcha-container, #ddos-protection,'
  + ' form[action*="captcha"]';
let marker = false;
try { marker = !!document.querySelector(sel); } catch (e) {}
const body = document.body ? (document.body.innerText || '') : '';
return [document.title || '', body.slice(0, 3000), marker];
"""


def challenge_reason(*, title: str, text: str, marker: bool) -> Optional[str]:
    """Why a page looks like a bot check, or None. A challenge title decides alone; a challenge phrase or widget
    counts only on a short page."""
    t = (title or "").strip().lower()
    for pattern in CHALLENGE_TITLES:
        if pattern in t:
            return f"title: {pattern!r}"
    body = " ".join((text or "").lower().split())
    if len(body) >= CHALLENGE_MAX_TEXT:
        return None
    if marker:
        return "bot-check widget on a short page"
    for phrase in CHALLENGE_PHRASES:
        if phrase in body:
            return f"text: {phrase!r}"
    return None


def load_webvoyager(*, project_root: str):
    """Import WebVoyager's run.py script (which imports its siblings by bare name) as `webvoyager_run`."""
    # Locked: a second thread must not see the module in sys.modules half-executed.
    with _LOAD_LOCK:
        if "webvoyager_run" in sys.modules:
            return sys.modules["webvoyager_run"]
        wv_dir = os.path.join(project_root, "WebVoyager")
        if wv_dir not in sys.path:
            sys.path.insert(0, wv_dir)
        spec = importlib.util.spec_from_file_location("webvoyager_run", os.path.join(wv_dir, "run.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["webvoyager_run"] = module
        return module


def action_menu(*, system_prompt: str, mode: str) -> str:
    """The action list from WebVoyager's system prompt; free_play drops ANSWER."""
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


def excluded_sites(*, parameters: dict) -> list:
    """The WebVoyager web_name values excluded from train and test."""
    return [s.strip() for s in str(parameters.get("web_excluded_sites") or "").split(",") if s.strip()]


def filtered_task_file(*, parameters: dict) -> str:
    """Writes WebVoyager's task file without the excluded sites; returns its path."""
    excluded = set(excluded_sites(parameters=parameters))
    from cusi.utils.paths import web_source_task_file, web_task_file
    source = web_source_task_file(parameters=parameters)
    with open(source) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    unknown = excluded - {r["web_name"] for r in rows}
    if unknown:
        raise ValueError(f"web_excluded_sites names sites not in {source}: {sorted(unknown)}")
    path = web_task_file(parameters=parameters)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            if row["web_name"] not in excluded:
                f.write(json.dumps(row) + "\n")
    return path


def load_task(*, task_id: str, data_file: str) -> dict:
    with open(data_file) as f:
        for line in f:
            task = json.loads(line)
            if task["id"] == task_id:
                return task
    raise KeyError(f"No WebVoyager task {task_id!r} in {data_file}")


class WebVoyagerPlayEnv(TextActionEnv):
    """A WebVoyager start page as a text-action gym.Env."""

    env_description = "a web browser"

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
        """url is free_play only; fast_waits replaces run.py's fixed sleeps process-wide (run.FAST_WAITS)."""
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
        self._raw_frame = None
        self._challenge = None      # why the current page looks like a bot check (challenge_reason), or None
        self._url = self.start_url
        self._states: dict[str, dict] = {}   # saved state_id -> browser state
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
        shutil.rmtree(self._download_dir, ignore_errors=True)
        os.makedirs(self._download_dir)
        self._args.chrome_profile_dir = profile
        # Simultaneous Chrome starts can fail ("DevToolsActivePort file doesn't exist"): serialise and retry.
        with _BROWSER_START_LOCK:
            for attempt in range(3):
                shutil.rmtree(profile, ignore_errors=True)
                options = self._wv.driver_config(self._args)
                try:
                    self._driver = self._wv._make_driver(self._args, options)
                    break
                except self._wv.WebDriverException as e:
                    if attempt == 2:
                        raise
                    log_warn(f"Chrome failed to start ({str(e).splitlines()[0][:120]}); retrying",
                             parameters=self._parameters)
                    time.sleep(3)
        self._download_files = []

    def _quit(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def _open(self, *, url: str) -> np.ndarray:
        """Fresh browser on url; returns the first frame."""
        self._new_browser()
        self._wv.open_start_page(self._driver, url)
        return self._observe()[0]

    def _observe(self):
        """Set-of-mark screenshot and element text as run.py builds them, plus the unlabelled raw frame."""
        rects, web_eles, web_eles_text = self._wv.get_web_element_rect(self._driver,
                                                                        fix_color=self._args.fix_box_color)
        png = self._driver.get_screenshot_as_png()
        try:
            for rect in rects:
                self._driver.execute_script("arguments[0].remove()", rect)
        except self._wv.StaleElementReferenceException:
            pass
        frame = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)
        try:
            raw_png = self._driver.get_screenshot_as_png()
            self._raw_frame = np.asarray(Image.open(io.BytesIO(raw_png)).convert("RGB"), dtype=np.uint8)
        except self._wv.WebDriverException:
            self._raw_frame = frame
        self._web_eles = web_eles
        try:
            self._url = self._driver.current_url
        except Exception:
            pass
        try:
            title, text, marker = self._driver.execute_script(_CHALLENGE_JS)
            self._challenge = challenge_reason(title=title, text=text, marker=bool(marker))
        except self._wv.WebDriverException:
            self._challenge = None
        self._last_obs = (frame, web_eles_text)
        return self._last_obs

    @property
    def raw_frame(self):
        return self._raw_frame

    def _obs(self) -> dict:
        frame, web_eles_text = self._last_obs
        return {"frame": frame, "texts": {"web_elements": web_eles_text},
                "actions": self.actions_text, "goal": self._goal}

    def sample_action(self, *, rng=None) -> str:
        """Click a labelled element or scroll the window; never actions that leave the scene or need content."""
        rng = self.np_random if rng is None else rng
        if self._web_eles and rng.random() < 0.75:
            return f"Thought: random action.\nAction: Click [{int(rng.integers(len(self._web_eles)))}]"
        return f"Thought: random action.\nAction: Scroll [WINDOW]; [{rng.choice(['up', 'down'])}]"

    def _on_pdf(self, pdf_path: str) -> str:
        shutil.copy(pdf_path, self._work_dir)
        return f"You downloaded a PDF file: {os.path.basename(pdf_path)}."

    # ---------------------------------------------------------------- gym API

    def _page_info(self, *, info: dict) -> dict:
        info["raw_frame"] = self._raw_frame
        info["challenge"] = self._challenge is not None
        if self._challenge is not None:
            info["challenge_reason"] = self._challenge
        return info

    def _reset_impl(self):
        obs, info = self._reset_core()
        return obs, self._page_info(info=info)

    def _step_impl(self, action: str):
        obs, reward, terminated, truncated, info = self._step_core(action)
        return obs, reward, terminated, truncated, self._page_info(info=info)

    def _reset_core(self):
        if not self._at_initial_state:
            self._open(url=self.start_url)
        self._at_initial_state = False
        self._steps = 0
        info = self.make_info(valid=True, error=None, parsed_action=None, step=0, scene_id=self.scene_id,
                              url=self._url)
        return self._obs(), info

    def _step_core(self, action: str):
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
                # As run.py: no re-observation; the agent retries on the same labels.
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

    # ------------------------------------------------------------ saved states
    # A saved state is URL, cookies, origin storage and scroll; in-page state and other tabs are not restored.

    # Pages without storage access (e.g. some error pages) throw; they save/restore none.
    _STORAGE_JS = ("try { return {local: Object.assign({}, window.localStorage),"
                   " session: Object.assign({}, window.sessionStorage)}; }"
                   " catch (e) { return {local: {}, session: {}}; }")
    _SET_STORAGE_JS = ("const s = arguments[0]; try {"
                       " window.localStorage.clear(); window.sessionStorage.clear();"
                       " for (const [k, v] of Object.entries(s.local)) window.localStorage.setItem(k, v);"
                       " for (const [k, v] of Object.entries(s.session)) window.sessionStorage.setItem(k, v);"
                       " } catch (e) {}")
    # Fields of a CDP Network.Cookie that Storage.setCookies accepts as a CookieParam.
    _COOKIE_PARAM_FIELDS = ("name", "value", "domain", "path", "secure", "httpOnly", "sameSite",
                            "expires", "priority", "sourceScheme", "sourcePort", "partitionKey")

    def _save_state_impl(self, *, state_id: str) -> None:
        try:
            cookies = self._driver.execute_cdp_cmd("Storage.getCookies", {})["cookies"]
            storage = self._driver.execute_script(self._STORAGE_JS)
            scroll = self._driver.execute_script("return [window.scrollX, window.scrollY];")
            url = self._driver.current_url
        except self._wv.WebDriverException as e:
            log_error(f"Could not save web state {state_id}: {str(e)[:200]}", parameters=self._parameters)
        self._states[state_id] = {"url": url, "cookies": cookies, "storage": storage, "scroll": scroll}

    def _load_state_impl(self, *, state_id: str) -> dict:
        state = self._states[state_id]
        cookies = []
        for cookie in state["cookies"]:
            param = {k: cookie[k] for k in self._COOKIE_PARAM_FIELDS if k in cookie}
            if cookie.get("session") or param.get("expires", -1) < 0:
                param.pop("expires", None)
            cookies.append(param)
        try:
            self._new_browser()
            self._driver.execute_cdp_cmd("Storage.setCookies", {"cookies": cookies})
            # Open the origin once to write storage, then reopen so page scripts start with it.
            self._driver.get(state["url"])
            self._driver.execute_script(self._SET_STORAGE_JS, state["storage"])
            self._wv.open_start_page(self._driver, state["url"])
            self._wait_for_stable_layout()
            self._driver.execute_script("window.scrollTo(arguments[0], arguments[1]);", *state["scroll"])
            self._wv._settle(self._driver, 2)
            self._observe()
        except self._wv.WebDriverException as e:
            log_error(f"Could not load web state {state_id}: {str(e)[:200]}", parameters=self._parameters)
        self._at_initial_state = False
        self._steps = 0
        return self._obs()

    def _wait_for_stable_layout(self, *, timeout: float = 8.0, stable_for: float = 1.0) -> None:
        """Wait until the page height stops changing, so the restored scroll lands where it was saved."""
        deadline = time.time() + timeout
        last, since = None, time.time()
        while time.time() < deadline:
            height = self._driver.execute_script("return document.documentElement.scrollHeight;")
            if height != last:
                last, since = height, time.time()
            elif time.time() - since >= stable_for:
                return
            time.sleep(0.2)

    def _delete_state_impl(self, *, state_id: str) -> None:
        del self._states[state_id]

    def _export_state_impl(self, *, state_id: str, directory: str) -> str:
        name = f"{state_id}.json"
        with open(os.path.join(directory, name), "w") as f:
            json.dump(self._states[state_id], f)
        return name

    def _import_state_impl(self, *, state_id: str, path: str) -> None:
        with open(path) as f:
            self._states[state_id] = json.load(f)

    def close(self) -> None:
        self._states.clear()
        self._saved_state_ids.clear()
        self._quit()
        shutil.rmtree(self._work_dir, ignore_errors=True)
