"""
WebVoyager as a TextActionEnv, driving the browser with WebVoyager's own run.py code.
Sites are live, so reset restores the browser, not the page content. Reward is 0: the LLM judge scores afterwards.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from urllib.parse import urlparse
import time
import zlib
from typing import Any, Optional
import numpy as np
from PIL import Image
from selenium.common.exceptions import JavascriptException, TimeoutException as ScriptTimeout
from urllib3.exceptions import HTTPError as WebDriverHang
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info, log_warn, log_error
from cusi.envs.base import TextActionEnv, scene_hash, screen_changed
from cusi.envs.web_boxes import MAX_BOX_CHARS, TEXT_BOXES_JS, web_text_boxes

TEXT_KEYS = ("web_elements",)
_BROWSER_START_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
DEFAULT_TEST_MAX_STEPS = 15   # WebVoyager/run.sh --max_iter

# run.py's format error, verbatim.
MSG_FORMAT = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
MSG_NO_ANSWER = "The ANSWER action is not available: there is no task to answer. No action was performed."
# Google is not an action of this env (decisions.md E8.2b, F14=google: it often lands on a CAPTCHA). The menu below
# and the agent's system prompt both drop it through without_google, and a Google action that arrives is rejected.
MSG_NO_GOOGLE = "The Google action is not available. Use the website's own search and navigation."
MSG_NO_LABELS = ("No element on this page has a numerical label, so Click and Type cannot be used here. "
                 "Use Scroll, GoBack or Wait.")
MSG_OUT_OF_BOUNDS = ("That action left the website for a browser page ({url}), so it was undone. "
                     "Stay on the website.")
MSG_OFF_SITE = "That action left {site} for another website ({url}), so it was undone. Stay on {site}."
#: An action that would land on a blocked website (an env server's held-out sites): undone before it is observed, and
#: the message does not name the site.
MSG_BLOCKED_SITE = "That action is not available here, so it was undone. No action was performed."
#: Typing mode (the typing explorer): only Type (which also presses Enter) and Scroll.
MSG_TYPING_ONLY = "Only Type and Scroll are available here. No action was performed."
#: No-typing mode (the VLM explorer without typing): Type is refused, and the menu and the agent's system prompt drop it
#: through without_typing.
MSG_NO_TYPING = "Typing is not available here. Click or scroll instead. No action was performed."
#: Browser-internal pages (the new-tab page after GoBack from the first page, settings, ...) are out of bounds:
#: no element on them can be labelled, and they are not website content.
OUT_OF_BOUNDS_PREFIXES = ("chrome://", "chrome-untrusted://", "chrome-search://")
# run.py's format_msg puts this sentence before the element list on every observation; the explorer prompt reuses it.
ELEMENTS_INTRO = ("I've provided the tag name of each element and the text it contains (if text exists). Note that "
                  "<textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and "
                  "then refer to the textual information.")


# --------------------------------------------------------------------------- bot-check pages
# A challenge page (Cloudflare, captcha, bot block) is not site content; the search scores it 0 and never expands it.

CHALLENGE_TITLES = ("just a moment", "attention required", "access denied", "are you a robot", "are you a human",
                    "robot or human", "security check", "captcha", "human verification", "pardon our interruption",
                    "verify you are human", "bot verification", "request blocked")
CHALLENGE_PHRASES = ("verify you are human", "verifying you are human", "checking your browser",
                     "checking if the site connection is secure", "unusual traffic", "are you a robot",
                     "press & hold", "press and hold", "complete the security check", "you have been blocked",
                     "enable javascript and cookies to continue", "pardon our interruption", "request unsuccessful",
                     "access denied", "not a robot", "solve the captcha", "complete the captcha",
                     "slide right to secure your access", "slide right to complete the puzzle")
#: A matched phrase or marker counts only on a page with less body text than this (a real page may mention them).
CHALLENGE_MAX_TEXT = 1500
# [title, first 3000 chars of the body text, whether a bot-check widget/iframe is in the DOM]
# Removes the set-of-mark overlays WebVoyager's get_web_element_rect appends to <body> (fixed, dashed outline,
# max z-index, no pointer events).
#: Each labelled element, if text can be typed into it (an enabled, writable <input> of a text type, <textarea> or
#: contenteditable), its identifying attributes, else null.
_FIELD_ATTRS_JS = """
var TEXT = ['', 'text', 'search', 'email', 'tel', 'password', 'url', 'number'];
return arguments[0].map(function (e) {
    try {
        var tag = e.tagName.toLowerCase(), type = (e.getAttribute('type') || '').toLowerCase();
        var ok = tag === 'textarea' || (tag === 'input' && TEXT.indexOf(type) >= 0) || e.isContentEditable;
        if (!ok || e.disabled || e.readOnly) return null;
        var label = '';
        if (e.labels && e.labels.length) label = e.labels[0].innerText;
        else if (e.getAttribute('aria-labelledby')) {
            var l = document.getElementById(e.getAttribute('aria-labelledby')); if (l) label = l.innerText;
        }
        var g = function (a) { return e.getAttribute(a) || ''; };
        return {tag: tag, type: type, id: g('id'), name: g('name'), placeholder: g('placeholder'),
                aria_label: g('aria-label'), title: g('title'), autocomplete: g('autocomplete'), role: g('role'),
                label: (label || '').trim().slice(0, 80),
                form_action: e.form ? (e.form.getAttribute('action') || '') : '',
                value: (e.value || e.innerText || '').slice(0, 80)};
    } catch (err) { return null; }
});
"""
#: Each labelled element's link target (its own href, or its enclosing link's), or null.
_ELEMENT_HREFS_JS = """
return arguments[0].map(function (e) {
    try {
        var a = (e.tagName === 'A' || e.tagName === 'AREA') ? e : (e.closest ? e.closest('a[href]') : null);
        return a && a.href ? a.href : null;
    } catch (err) { return null; }
});
"""
#: Each labelled element as [tag, its text (or aria-label, placeholder; first 60 chars), its value (a field's
#: content, first 60 chars), x, y, width, height] (viewport pixels, rounded): what identifies "element i" when a peek
#: replays actions in another browser, and what a synced sandbox must show.
_ELEMENT_DESC_JS = """
return arguments[0].map(function (e) {
    try {
        var r = e.getBoundingClientRect();
        var t = (e.innerText || e.getAttribute('aria-label') || e.getAttribute('placeholder') || '');
        var v = (typeof e.value === 'string') ? e.value : '';
        return [e.tagName.toLowerCase(), t.trim().replace(/\\s+/g, ' ').slice(0, 60), v.slice(0, 60),
                Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)];
    } catch (err) { return null; }
});
"""
_CLEAR_LABELS_JS = """
for (const el of Array.from(document.body ? document.body.children : [])) {
  const st = el.style;
  if (st && el.tagName === 'DIV' && st.position === 'fixed' && st.pointerEvents === 'none'
      && st.zIndex === '2147483647' && st.outline.includes('dashed')) el.remove();
}
"""
_CHALLENGE_JS = """
const sel = '#challenge-form, #challenge-running, #challenge-stage, #cf-challenge-running, .cf-turnstile,'
  + ' [name="cf-turnstile-response"], iframe[src*="challenges.cloudflare.com"], iframe[src*="hcaptcha.com"],'
  + ' iframe[src*="recaptcha"], .h-captcha, .g-recaptcha, #px-captcha, #captcha-container, #ddos-protection,'
  + ' form[action*="captcha"], iframe[src*="captcha-delivery.com"]';     // DataDome
let marker = false;
try { marker = !!document.querySelector(sel); } catch (e) {}
const body = document.body ? (document.body.innerText || '') : '';
return [document.title || '', body.slice(0, 3000), marker];
"""

# true_state: [title, a hash of the page's skeleton (the set of tag paths below <body>, to SKELETON_DEPTH, without
# text, scripts or repeats), the number of paths]. Pages of one kind (two arXiv abstracts) share a skeleton.
SKELETON_DEPTH = 8
_SKELETON_JS = """
const paths = new Set(), skip = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'svg', 'SVG']);
const limit = arguments[0];
(function rec(el, path, depth) {
  if (depth >= limit) return;
  for (const c of el.children) {
    if (skip.has(c.tagName)) continue;
    const p = path + '>' + c.tagName.toLowerCase();
    paths.add(p);
    rec(c, p, depth + 1);
  }
})(document.body || document.documentElement, 'body', 0);
const s = Array.from(paths).sort().join('|');
let h = 2166136261;
for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
return [document.title || '', h.toString(16), paths.size];
"""


# Client-side limit on one WebDriver call. A page whose renderer freezes otherwise blocks chromedriver, and the search,
# indefinitely (seen: a 20+ minute hang in execute_script); the call then raises WebDriverHang and the browser is
# restarted like a dead one.
WEBDRIVER_TIMEOUT = 120


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


def elements_text(*, web_eles_text: str, n_labels: int) -> str:
    """The element list WebVoyager's agent is shown after ELEMENTS_INTRO (run.py format_msg's web_text); on a page
    with no labelled element, a note that Click and Type cannot be used."""
    return web_eles_text if n_labels else MSG_NO_LABELS


def out_of_bounds(*, url: Optional[str]) -> bool:
    return bool(url) and url.startswith(OUT_OF_BOUNDS_PREFIXES)


def in_scope(*, url: Optional[str], scope: Optional[dict]) -> bool:
    """Whether url is on a scene's website: scope {"hosts": [domain, ...], "path": optional prefix} admits each
    domain and its subdomains (support.apple.com for apple.com), and with "path" only URLs under it (Google Flights and
    Google Maps share www.google.com). No scope, no url, or a non-web URL (about:, data:) counts as in scope."""
    if scope is None or not url:
        return True
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        return True
    host = (parts.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in scope["hosts"]):
        return False
    return not scope.get("path") or parts.path.startswith(scope["path"])


def same_page_url(*, a: Optional[str], b: Optional[str]) -> bool:
    """Whether two URLs name the same page: equal scheme, host, path and query keys. Query values may differ (a
    site's per-load tokens, Booking's chal_t), and so may the path segments of a map's viewport (Google Maps'
    /@lat,lng,zoom and /data=...): what they would change shows in the page itself."""
    if a == b:
        return True
    if not a or not b:
        return False

    def key(url):
        p = urlparse(url)
        path = "/".join(seg for seg in p.path.split("/") if not seg.startswith(("@", "data=")))
        return p.scheme, p.netloc, path, sorted(kv.split("=", 1)[0] for kv in p.query.split("&") if kv)
    return key(a) == key(b)


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


def remove_line(*, text: str, pattern: str) -> str:
    """text without the one line matching pattern; raises if there is not exactly one (the submodule text changed)."""
    out, n = re.subn(pattern, "", text)
    if n != 1:
        raise ValueError(f"WebVoyager system prompt anchor matched {n} times (WebVoyager changed?): {pattern!r}")
    return out


def without_google(*, prompt: str) -> str:
    """WebVoyager's system prompt without its Google action (the numbered entry and the format line)."""
    prompt = remove_line(text=prompt, pattern=r"6\. Google, directly jump[^\n]*\n")
    return remove_line(text=prompt, pattern=r"- Google\n")


def without_typing(*, prompt: str) -> str:
    """WebVoyager's system prompt without typing: the Type action (the numbered entry and the format line) and the two
    action guidelines about typing."""
    prompt = remove_line(text=prompt, pattern=r"2\. Delete existing content in a textbox[^\n]*\n")
    prompt = remove_line(text=prompt, pattern=r"- Type \[Numerical_Label\]; \[Content\]\n")
    prompt = remove_line(text=prompt, pattern=r"1\) To input text, NO need to click textbox first[^\n]*\n")
    return remove_line(text=prompt, pattern=r"2\) You must Distinguish between textbox and search button[^\n]*\n")


def action_menu(*, system_prompt: str, mode: str, action_mode: str = "all") -> str:
    """The action list from WebVoyager's system prompt, without Google; free_play drops ANSWER, action_mode
    "no_typing" drops Type."""
    system_prompt = without_google(prompt=system_prompt)
    if action_mode == "no_typing":
        system_prompt = without_typing(prompt=system_prompt)
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
    text = "".join(json.dumps(row) + "\n" for row in rows if row["web_name"] not in excluded)
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == text:
                return path
    # Concurrent evals call this too: write to a private file and rename, so a reader never sees a partial file.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
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
    elements_intro = ELEMENTS_INTRO      # the VLM explorer shows it before obs["elements"], as the eval agent does

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
        scope: Optional[dict] = None,
        site: Optional[str] = None,
        blocked: Optional[list] = None,
        action_mode: str = "all",
        parameters: dict[str, Any] = None,
    ) -> None:
        """url is free_play only; fast_waits replaces run.py's fixed sleeps process-wide (run.FAST_WAITS). scope (see
        in_scope) keeps a free_play episode on its website: a step that lands elsewhere is undone and invalid; site
        names it in that message. Test mode is never scoped (a few benchmark tasks need another site). blocked (any mode): a
        list of scopes (an env server's held-out websites); a step that lands on one is undone before it is observed
        and is invalid with MSG_BLOCKED_SITE. action_mode
        "typing" accepts only Type and Scroll; "no_typing" refuses Type. A Type that leaves the page's URL unchanged (typed, not submitted to a
        new page) sets info["no_node"]: a saved state (URL, cookies, storage, scroll) would not hold what was typed."""
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
        self._text_boxes = []
        self._challenge = None      # why the current page looks like a bot check (challenge_reason), or None
        self._url = self.start_url
        self._scope = scope if mode == "free_play" else None
        self._blocked = list(blocked or [])
        if action_mode not in ("all", "typing", "no_typing"):
            log_error(f"action_mode must be 'all', 'typing' or 'no_typing', got {action_mode!r}",
                      parameters=self._parameters)
        self.action_mode = action_mode
        self._site = site or urlparse(self.start_url).hostname
        self.site_name = self._site if self._scope is not None else None   # the explorer's "Do not leave" names it
        self._ele_off_site: list = []        # per labelled element: a link off the scope's website
        self._js_coverage = False
        self._js_seen: set = set()      # hashes of every (script, function) this env's browsers have run
        self._states: dict[str, dict] = {}   # saved state_id -> browser state
        self.browser_restarts = 0
        self._ele_desc: list = []        # per labelled element: _ELEMENT_DESC_JS's row (or None)
        # peek's sandbox browser (see _peek_impl): its values of _SANDBOX_ATTRS while the main browser's are in place.
        sandbox_dir = os.path.join(self._work_dir, "peek")
        self._sandbox = {"_driver": None, "_web_eles": [], "_last_obs": None, "_raw_frame": None, "_text_boxes": [],
                         "_challenge": None, "_url": None, "_ele_off_site": [], "_ele_desc": [], "_download_files": [],
                         "_work_dir": sandbox_dir, "_download_dir": os.path.join(sandbox_dir, "downloads"),
                         "_args": argparse.Namespace(**{**vars(self._args),
                                                        "download_dir": os.path.join(sandbox_dir, "downloads")}),
                         "_js_coverage": False}
        self._in_sandbox = False
        # The episode so far, for the sandbox to reproduce: where it started ({"open": url} or {"state": saved state})
        # and every step since, each {"action", "desc" (the element rows the action's index refers to), "new_doc"
        # (whether the step loaded a new document), "snapshot" (the browser state after a new_doc step, or None)}.
        self._episode_log = {"start": {"open": self.start_url}, "steps": []}
        self._doc_id = None

        frame = self._open(url=self.start_url)
        self._doc_id = self._document_id()
        super().__init__(mode=mode, frame_shape=frame.shape, text_keys=TEXT_KEYS, parameters=self._parameters)
        if max_steps is None and mode == "test":
            max_steps = DEFAULT_TEST_MAX_STEPS
        self.max_steps = max_steps
        self.actions_text = action_menu(system_prompt=self._wv.SYSTEM_PROMPT, mode=mode, action_mode=action_mode)
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
        from selenium.webdriver.remote.remote_connection import RemoteConnection
        with _BROWSER_START_LOCK:
            for attempt in range(3):
                shutil.rmtree(profile, ignore_errors=True)
                options = self._wv.driver_config(self._args)
                RemoteConnection.set_timeout(WEBDRIVER_TIMEOUT)   # read once, when the driver's connection is made
                try:
                    self._driver = self._wv._make_driver(self._args, options)
                    break
                except self._wv.WebDriverException as e:
                    if attempt == 2:
                        raise
                    log_warn(f"Chrome failed to start ({str(e).splitlines()[0][:120]}); retrying",
                             parameters=self._parameters)
                    time.sleep(3)
                finally:
                    RemoteConnection.reset_timeout()
        self._download_files = []
        # true_state's JavaScript coverage: which functions the site's scripts have run (DevTools precise coverage).
        try:
            self._driver.execute_cdp_cmd("Profiler.enable", {})
            self._driver.execute_cdp_cmd("Profiler.startPreciseCoverage", {"callCount": False, "detailed": False})
            self._js_coverage = True
        except Exception as e:
            log_warn(f"JavaScript coverage unavailable ({str(e).splitlines()[0][:120]})", parameters=self._parameters)
            self._js_coverage = False

    def _quit(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    def _open(self, *, url: str, attempts: int = 3) -> np.ndarray:
        """Fresh browser on url; returns the first frame. A heavy page on a busy node can time out its first screenshot
        (Apple's start page did, twice, with three searches' 30 browsers on one node): retried in a new browser."""
        for attempt in range(1, attempts + 1):
            try:
                self._new_browser()
                self._wv.open_start_page(self._driver, url)
                return self._observe()[0]
            except (self._wv.WebDriverException, WebDriverHang) as e:
                if attempt == attempts:
                    raise
                log_warn(f"opening {url} failed ({type(e).__name__}: {str(e)[:120]}); retry {attempt}/{attempts - 1} "
                         f"in a new browser", parameters=self._parameters)

    def _observe(self):
        """Set-of-mark screenshot and element text as run.py builds them, plus the unlabelled raw frame."""
        for attempt in range(3):
            try:
                rects, web_eles, web_eles_text = self._wv.get_web_element_rect(self._driver,
                                                                                fix_color=self._args.fix_box_color)
                break
            except (self._wv.StaleElementReferenceException, JavascriptException, ScriptTimeout) as e:
                # The page changed while it was being labelled, or the labelling script fails or times out on it (e.g. a
                # non-HTML document, a page whose scripts keep it busy): drop the overlays already drawn and relabel. A page that still can't be labelled is
                # observed without element labels rather than ending the episode.
                try:
                    self._driver.execute_script(_CLEAR_LABELS_JS)
                except (JavascriptException, ScriptTimeout):
                    pass
                reason = str(e).splitlines()[0][:150]
                if attempt == 2:
                    log_warn(f"labelling failed 3 times ({reason}); observing the page without element labels",
                             parameters=self._parameters)
                    rects, web_eles, web_eles_text = [], [], ""
                    break
                log_warn(f"labelling failed ({reason}); retrying", parameters=self._parameters)
                time.sleep(1)
        for attempt in range(3):
            # A navigation that starts mid-screenshot aborts it ("aborted by navigation"): wait for the page and retry.
            try:
                png = self._driver.get_screenshot_as_png()
                break
            except self._wv.WebDriverException:
                if attempt == 2:
                    raise
                time.sleep(2)
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
        self._ele_desc = []
        if web_eles:
            try:
                self._ele_desc = self._driver.execute_script(_ELEMENT_DESC_JS, web_eles)
            except self._wv.WebDriverException:
                pass
        self._ele_off_site = []
        if self._scope is not None and web_eles:
            try:
                hrefs = self._driver.execute_script(_ELEMENT_HREFS_JS, web_eles)
                self._ele_off_site = [bool(h) and not in_scope(url=h, scope=self._scope) for h in hrefs]
            except self._wv.WebDriverException:
                pass
        try:
            dpr, rows = self._driver.execute_script(TEXT_BOXES_JS, web_eles, MAX_BOX_CHARS)
            h, w = self._raw_frame.shape[:2]
            self._text_boxes = web_text_boxes(rows=rows, dpr=float(dpr), width=w, height=h)
        except self._wv.WebDriverException as e:
            log_warn(f"text boxes failed: {str(e)[:200]}", parameters=self._parameters)
            self._text_boxes = []
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
        return {"frame": frame, "texts": {"web_elements": web_eles_text}, "text_boxes": list(self._text_boxes),
                "actions": self.actions_text, "goal": self._goal,
                "elements": elements_text(web_eles_text=web_eles_text, n_labels=len(self._web_eles))}

    def typable_fields(self) -> list:
        """The labelled elements text can be typed into on the current page (Type [index]), each {"index", "kind",
        "label", "value", "signature"}: signature holds what may identify the field across pages (page host and path,
        tag, type, id, name, placeholder, aria-label, label text, autocomplete, role, form action)."""
        if not self._web_eles:
            return []
        try:
            rows = self._driver.execute_script(_FIELD_ATTRS_JS, self._web_eles)
        except self._wv.WebDriverException:
            return []
        page = urlparse(self._url or "")
        fields = []
        for i, r in enumerate(rows or []):
            if not r:
                continue
            value = r.pop("value")
            label = next((r[k] for k in ("label", "aria_label", "placeholder", "title", "name", "id") if r[k]), "")
            fields.append({"index": i, "kind": f"{r['tag']}:{r['type']}" if r["type"] else r["tag"], "label": label,
                           "value": value, "signature": {"host": page.hostname, "path": page.path, **r}})
        return fields

    def sample_action(self, *, rng=None) -> str:
        """Click a labelled element or scroll the window; never actions that leave the scene or need content. With a
        scope, links to other websites are not clicked (they would only be undone)."""
        rng = self.np_random if rng is None else rng
        clickable = [i for i in range(len(self._web_eles))
                     if not (i < len(self._ele_off_site) and self._ele_off_site[i])]
        if clickable and rng.random() < 0.75:
            return f"Thought: random action.\nAction: Click [{int(clickable[int(rng.integers(len(clickable)))])}]"
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
        self._episode_log = {"start": {"open": self.start_url}, "steps": []}
        self._doc_id = self._document_id()
        return obs, self._page_info(info=info)

    def _step_impl(self, action: str):
        if self._in_sandbox:
            obs, reward, terminated, truncated, info = self._step_core(action)
            return obs, reward, terminated, truncated, self._page_info(info=info)
        record = {"action": action, "desc": list(self._ele_desc), "new_doc": False, "snapshot": None}
        obs, reward, terminated, truncated, info = self._step_core(action)
        # A step that loaded a new document (a navigation, not an in-page change or a pushState URL) is where the
        # sandbox can start reproducing the episode: keep the browser state it left.
        doc_id = self._document_id()
        if doc_id is None or doc_id != self._doc_id:
            record["new_doc"] = True
            try:
                record["snapshot"] = self._browser_state()
            except (self._wv.WebDriverException, WebDriverHang):
                pass
        self._doc_id = doc_id
        self._episode_log["steps"].append(record)
        return obs, reward, terminated, truncated, self._page_info(info=info)

    def _document_id(self) -> Optional[float]:
        """The current document's performance.timeOrigin: it changes when a new document loads and not on in-page
        changes (including history.pushState). None if it cannot be read."""
        try:
            return self._driver.execute_script("return performance.timeOrigin;")
        except (self._wv.WebDriverException, WebDriverHang):
            return None

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

        if action_key == "google":
            info = self.make_info(valid=False, error=MSG_NO_GOOGLE, parsed_action=parsed, step=self._steps,
                                  scene_id=self.scene_id, stale=True)
            return self._obs(), 0.0, False, truncated, info

        if self.action_mode == "no_typing" and action_key == "type":
            info = self.make_info(valid=False, error=MSG_NO_TYPING, parsed_action=parsed, step=self._steps,
                                  scene_id=self.scene_id, stale=True)
            return self._obs(), 0.0, False, truncated, info

        if self.action_mode == "typing" and action_key not in ("type", "scroll"):
            info = self.make_info(valid=False, error=MSG_TYPING_ONLY, parsed_action=parsed, step=self._steps,
                                  scene_id=self.scene_id, stale=True)
            return self._obs(), 0.0, False, truncated, info

        if action_key == "answer":
            if self.mode == "test":
                info = self.make_info(valid=True, error=None, parsed_action=parsed, step=self._steps,
                                      scene_id=self.scene_id, answer=parsed["text"], url=self._url)
                return self._obs(), 0.0, True, False, info
            info = self.make_info(valid=False, error=MSG_NO_ANSWER, parsed_action=parsed, step=self._steps,
                                  scene_id=self.scene_id, stale=True)
            return self._obs(), 0.0, False, truncated, info

        extra = {}
        prev_url = self._url
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
            self._close_other_tabs()
            self._observe()
            error = self._left_message(url=self._url) if self._left_message(url=prev_url) is None else None
            if error is not None:
                left = self._url
                self._return_to(url=prev_url)
                where = {"blocked" if error == MSG_BLOCKED_SITE else "out_of_bounds": left}
                info = self.make_info(valid=False, error=error, parsed_action=parsed, step=self._steps,
                                      scene_id=self.scene_id, url=self._url, **where)
                return self._obs(), 0.0, False, truncated, info
        except (self._wv.WebDriverException, WebDriverHang) as e:
            # A navigation that aborts a command can leave the tab detached ("not attached to an active page",
            # "inspected target navigated or closed"), and a hung renderer times out every command: either way every
            # retry fails, so restart Chrome as for a crash
            detached = any(s in str(e).lower() for s in ("aborted by navigation",
                                                         "timed out receiving message from renderer"))
            if not (isinstance(e, WebDriverHang) or self._wv._is_browser_dead(e) or detached):
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
        if action_key == "type" and self._url == prev_url:
            extra["no_node"] = "typed but not submitted to a new page"
        info = self.make_info(valid=True, error=None, parsed_action=parsed, step=self._steps,
                              scene_id=self.scene_id, url=self._url, **extra)
        return self._obs(), 0.0, False, truncated, info

    def _left_message(self, *, url: Optional[str]) -> Optional[str]:
        """The error for a step that landed on url, if url is a browser page or (scoped) another website."""
        if url and url.startswith("http") and any(in_scope(url=url, scope=b) for b in self._blocked):
            return MSG_BLOCKED_SITE
        if out_of_bounds(url=url):
            return MSG_OUT_OF_BOUNDS.format(url=url)
        if not in_scope(url=url, scope=self._scope):
            return MSG_OFF_SITE.format(site=self._site, url=url)
        return None

    def _return_to(self, *, url: str) -> None:
        """Undo a step that left the website: browser Back keeps the page's state (form inputs, a map's view); if Back
        does not land on the website (a redirect chain), load url."""
        try:
            self._driver.back()
            self._wait_for_stable_layout()
            back_url = self._driver.current_url
        except self._wv.WebDriverException:
            back_url = None
        if self._left_message(url=back_url) is not None:
            self._driver.get(url)
            self._wait_for_stable_layout()
        self._observe()

    def _close_other_tabs(self) -> None:
        """Keep one tab: clicks open links in place (run.py sets target=_self), but a page's script can still open
        another window, which no observation would show."""
        try:
            handles = self._driver.window_handles
            if len(handles) > 1:
                keep = self._driver.current_window_handle
                for h in handles:
                    if h != keep:
                        self._driver.switch_to.window(h)
                        self._driver.close()
                self._driver.switch_to.window(keep)
        except self._wv.WebDriverException:
            pass

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
            self._states[state_id] = self._browser_state()
        except (self._wv.WebDriverException, WebDriverHang) as e:
            log_error(f"Could not save web state {state_id}: {str(e)[:200]}", parameters=self._parameters)

    def _load_state_impl(self, *, state_id: str) -> dict:
        state = self._states[state_id]
        try:
            self._restore_browser_state(state=state)
        except (self._wv.WebDriverException, WebDriverHang) as e:
            log_error(f"Could not load web state {state_id}: {str(e)[:200]}", parameters=self._parameters)
        self._at_initial_state = False
        self._steps = 0
        self._episode_log = {"start": {"state": state}, "steps": []}
        self._doc_id = self._document_id()
        return self._obs()

    def _browser_state(self) -> dict:
        """The current browser's saved-state dict (URL, cookies, storage, scroll); raises on a WebDriver error."""
        cookies = self._driver.execute_cdp_cmd("Storage.getCookies", {})["cookies"]
        storage = self._driver.execute_script(self._STORAGE_JS)
        scroll = self._driver.execute_script("return [window.scrollX, window.scrollY];")
        return {"url": self._driver.current_url, "cookies": cookies, "storage": storage, "scroll": scroll}

    def _restore_browser_state(self, *, state: dict) -> None:
        """A new browser on a _browser_state dict, observed; raises on a WebDriver error."""
        cookies = []
        for cookie in state["cookies"]:
            param = {k: cookie[k] for k in self._COOKIE_PARAM_FIELDS if k in cookie}
            if cookie.get("session") or param.get("expires", -1) < 0:
                param.pop("expires", None)
            cookies.append(param)
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

    # ------------------------------------------------------------------ peek
    # The main browser is never touched: each candidate runs in a second, sandbox browser that is first brought to the
    # main browser's state by reproducing the episode (_sync_sandbox), and checked against it (_sync_check). Saved states
    # cannot do this exactly (in-page state: typed text, open pickers, filters, is not in them).

    #: The attributes that belong to one browser; _sandboxed swaps the sandbox's in for the main browser's.
    _SANDBOX_ATTRS = ("_driver", "_web_eles", "_last_obs", "_raw_frame", "_text_boxes", "_challenge", "_url",
                      "_ele_off_site", "_ele_desc", "_download_files", "_work_dir", "_download_dir", "_args",
                      "_js_coverage")
    #: Episode bookkeeping that the sandbox's steps change and peek puts back (the sandbox's coverage read included).
    _peek_keep = ("_steps", "_at_initial_state", "_js_seen", "browser_restarts")

    @contextlib.contextmanager
    def _sandboxed(self):
        """Inside, every browser method (_open, _observe, _step_core, true_state, ...) works on the sandbox browser."""
        main = {k: getattr(self, k) for k in self._SANDBOX_ATTRS}
        for k in self._SANDBOX_ATTRS:
            setattr(self, k, self._sandbox[k])
        self._in_sandbox = True
        try:
            yield
        finally:
            for k in self._SANDBOX_ATTRS:
                self._sandbox[k] = getattr(self, k)
                setattr(self, k, main[k])
            self._in_sandbox = False

    def _remap(self, *, action: str, desc: list) -> tuple:
        """action with its element index moved to the same element in the current (sandbox) labelling: desc is the
        element rows of the labelling the index was chosen on. Returns (action text, the new index or None if
        unchanged, an error or None). The same element is the one with equal rows, else the one with the same tag and
        text nearest to where it was."""
        key, parsed = self._wv.extract_information(action_text(text=action))
        if key == "click":
            index = int(parsed[0])
        elif key in ("type", "scroll") and parsed["number"] != "WINDOW":
            index = int(parsed["number"])
        else:
            return action, None, None
        here = self._ele_desc
        if here == desc or (index < len(here) and index < len(desc) and here[index] == desc[index]):
            return action, None, None
        if index >= len(desc) or not desc[index]:
            return action, None, None        # an index that was out of range stays so (the step reports it)
        same = [j for j, d in enumerate(here) if d and d[:2] == desc[index][:2]]
        if not same:
            return action, None, f"element {index} {desc[index][:2]} not on the page"
        x, y = desc[index][3], desc[index][4]
        j = min(same, key=lambda j: abs(here[j][3] - x) + abs(here[j][4] - y))
        if key == "click":
            return f"Click [{j}]", j, None
        if key == "type":
            return f"Type [{j}]; [{parsed['content']}]", j, None
        return f"Scroll [{j}]; [{parsed['content']}]", j, None

    def _sync_methods(self, *, action: str) -> list:
        """The ways to sync the sandbox for a candidate, most faithful first (see _sync_sandbox). A GoBack, as the
        candidate or among the steps an anchor replay would replay, needs the browser's history, which only the full
        replay rebuilds: it is synced only so."""
        steps = self._episode_log["steps"]
        anchor = self._sync_anchor()
        if self._is_goback(action=action) or (anchor is not None and any(
                self._is_goback(action=r["action"]) for r in steps[anchor + 1:])):
            return ["full"]
        if not steps:
            return ["full"]
        return (["anchor"] if anchor is not None else []) + ["full", "state"]

    def _sync_anchor(self) -> Optional[int]:
        """The last logged step that loaded a new document and kept its browser state, or None."""
        steps = self._episode_log["steps"]
        return next((i for i in range(len(steps) - 1, -1, -1) if steps[i]["new_doc"] and steps[i]["snapshot"]), None)

    def _sync_sandbox(self, *, method: str, ref: dict) -> dict:
        """Bring the (current, sandbox) browser to the main episode's state. "anchor": open the browser state saved
        after the last step that loaded a new document, then replay the steps since (in-page: typing, clicks that did
        not navigate, scrolls, pushState URLs); "full": open the episode's start and replay every step; "state": open
        the main browser's current URL, cookies, storage and scroll (ref["state"]; in-page state is lost, as with
        load_state). Replayed element indices are remapped (_remap). Returns {"method", "replayed", "remapped",
        "error"}."""
        steps = self._episode_log["steps"]
        out = {"method": method, "replayed": 0, "remapped": 0, "error": None}
        if method == "state":
            self._restore_browser_state(state=ref["state"])
            return out
        if method == "anchor":
            anchor = self._sync_anchor()
            self._restore_browser_state(state=steps[anchor]["snapshot"])
            todo = steps[anchor + 1:]
        else:
            start = self._episode_log["start"]
            if "open" in start:
                self._open(url=start["open"])
            else:
                self._restore_browser_state(state=start["state"])
            todo = steps
        for record in todo:
            action, index, error = self._remap(action=record["action"], desc=record["desc"])
            if error is not None:
                out["error"] = f"replay step {out['replayed'] + 1}: {error}"
            out["remapped"] += index is not None
            self._step_core(action)
            out["replayed"] += 1
        return out

    def _sync_check(self, *, ref: dict) -> dict:
        """The (sandbox) browser against the main browser's reference (_peek_ref): which of URL, element rows, element
        text, skeleton and screen agree, and the screen's mean absolute pixel difference. "synced": the same page
        (same_page_url) and element rows (what the agent is shown; the skeleton also differs by hidden nodes, a site's
        late widgets, and the screen by animation, so they are reported, not required). On a URL that differs,
        "sandbox_url" is the sandbox's."""
        try:
            title, skeleton, _ = self._driver.execute_script(_SKELETON_JS, SKELETON_DEPTH)
        except (self._wv.WebDriverException, WebDriverHang):
            title, skeleton = None, None
        a, b = ref["raw_frame"], self._raw_frame
        same_shape = a is not None and b is not None and a.shape == b.shape
        out = {"url": self._url == ref["url"], "elements": self._ele_desc == ref["desc"],
               "elements_text": self._last_obs[1] == ref["elements_text"], "skeleton": skeleton == ref["skeleton"],
               "title": title == ref["title"],
               "screen": not screen_changed(before=a, after=b, ignore_top=self.change_ignore_top),
               "mae": float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) if same_shape else None}
        out["synced"] = same_page_url(a=self._url, b=ref["url"]) and out["elements"]
        out["score"] = (out["synced"], same_page_url(a=self._url, b=ref["url"]), out["elements"], out["skeleton"],
                        out["elements_text"], out["screen"], -(out["mae"] if out["mae"] is not None else 255.0))
        if not out["url"]:
            out["sandbox_url"] = self._url
        return out

    def _peek_ref(self) -> dict:
        """What the sandbox must match: the main browser's last observation, and its skeleton (read, not changed)."""
        try:
            title, skeleton, _ = self._driver.execute_script(_SKELETON_JS, SKELETON_DEPTH)
        except (self._wv.WebDriverException, WebDriverHang):
            title, skeleton = None, None
        try:
            state = self._browser_state() if self._episode_log["steps"] else None
        except (self._wv.WebDriverException, WebDriverHang):
            state = None
        return {"url": self._url, "desc": list(self._ele_desc), "elements_text": self._last_obs[1],
                "raw_frame": self._raw_frame, "title": title, "skeleton": skeleton, "state": state}

    def _is_goback(self, *, action: str) -> bool:
        return self._wv.extract_information(action_text(text=action))[0] == "goback"

    def _needs_browser(self, *, action: str) -> bool:
        """Whether _step_core would act on the browser (not reject or answer the action first)."""
        key, _ = self._wv.extract_information(action_text(text=action))
        if key in (None, "google", "answer"):
            return False
        if self.action_mode == "no_typing" and key == "type":
            return False
        return not (self.action_mode == "typing" and key not in ("type", "scroll"))

    def _peek_impl(self, *, actions: list) -> list:
        """Each candidate in the sandbox browser, synced to the main state just before it (_sync_sandbox: from the
        last new document, else the whole episode replayed, else the current URL, cookies and storage; the first that
        matches the main state, _sync_check). Actions rejected before reaching the browser
        are stepped on the main env, which they do not change. Extra info per candidate: "peek_sync" (_sync_check of
        the synced sandbox, with _sync_sandbox's method / replayed / remapped / error and the methods "tried"),
        "peek_synced", "peek_index"
        (the sandbox's index for the candidate's element when the labelling differed) and, if the sandbox failed or
        did not show the candidate's element, "peek_failed" (the result is then the current observation, invalid).
        The main browser only has its state read (_peek_ref)."""
        ref = self._peek_ref()
        steps = self._steps
        results = []
        for action in actions:
            start = time.time()
            self._steps = steps
            if not self._needs_browser(action=action):
                # Not _step_impl, which would log the step, nor true_state, which would take the main browser's
                # coverage away from its next step.
                obs, reward, terminated, truncated, info = self._step_core(action)
                self._page_info(info=info)
                info.update(changed=False, peek_synced=True, peek_sync={"method": "none"},
                            true=self._true_state(coverage=False))
            else:
                obs, reward, terminated, truncated, info = self._peek_in_sandbox(action=action, ref=ref, steps=steps)
            info["peek_seconds"] = time.time() - start
            info["peek_restored"] = True        # the main browser was not used
            results.append((obs, reward, terminated, truncated, info))
        self._steps = steps
        return results

    def _peek_in_sandbox(self, *, action: str, ref: dict, steps: int) -> tuple:
        try:
            with self._sandboxed():
                # The first method whose sandbox matches the main state; if none does, the closest one once more (a
                # live page can load differently twice), kept whether it then matches or not.
                methods = [m for m in self._sync_methods(action=action) if m != "state" or ref["state"] is not None]
                tried = []
                for method in methods:
                    sync = self._sync_sandbox(method=method, ref=ref)
                    check = self._sync_check(ref=ref)
                    tried.append((check["score"], method))
                    if check["synced"]:
                        break
                if not check["synced"]:
                    method = max(tried)[1]
                    sync = self._sync_sandbox(method=method, ref=ref)
                    check = self._sync_check(ref=ref)
                    tried.append((check["score"], method))
                check.update(sync)
                check["tried"] = [m for _, m in tried]
                del check["score"]
                text, index, error = self._remap(action=action, desc=ref["desc"])
                if error is None:
                    self.true_state()        # drains the coverage the sync ran, so js_new is the candidate's alone
                    before = self._raw_frame
                    self._steps = steps
                    obs, reward, terminated, truncated, info = self._step_impl(text)
                    self._check_texts(obs=obs)
                    info["changed"] = screen_changed(before=before, after=self._raw_frame,
                                                     ignore_top=self.change_ignore_top)
                    self._add_true_state(info=info)
        except (self._wv.WebDriverException, WebDriverHang) as e:
            log_warn(f"peek of {action_text(text=action)!r} failed in the sandbox browser: {str(e)[:200]}",
                     parameters=self._parameters)
            return self._peek_failed(failure=str(e)[:200], steps=steps, sync=None)
        if error is not None:
            # The candidate's element is not on the sandbox's page: stepping its index there would act on another
            # element, so there is no result.
            return self._peek_failed(failure=f"candidate {error}", steps=steps, sync=check)
        if index is not None:
            info["peek_index"] = index
            if info.get("parsed_action") and "index" in info["parsed_action"]:
                key, parsed = self._wv.extract_information(action_text(text=action))
                info["parsed_action"]["index"] = canonical_action(action_key=key, info=parsed)["index"]
        info["peek_sync"] = check
        info["peek_synced"] = check["synced"]
        return obs, reward, terminated, truncated, info

    def _peek_failed(self, *, failure: str, steps: int, sync: Optional[dict]) -> tuple:
        """A candidate the sandbox could not run: the current (main) observation, invalid, with info["peek_failed"]."""
        info = self.make_info(valid=False, error="The peek failed.", parsed_action=None, step=steps + 1,
                              scene_id=self.scene_id, stale=True, peek_failed=failure, peek_synced=False,
                              peek_sync=sync or {})
        self._page_info(info=info)
        info["true"] = self._true_state(coverage=False)
        return self._obs(), 0.0, False, False, info

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

    def true_state(self) -> Optional[dict]:
        """The URL, title and page skeleton hash, and js_new: how many JavaScript functions (script URL without its
        query, function name, offset) ran for the first time in this env during the last step."""
        return self._true_state(coverage=True)

    def _true_state(self, *, coverage: bool) -> dict:
        """true_state; without coverage, no js_new and the browser's coverage is left unread."""
        out = {"url": self._url}
        if self._driver is None:
            return out
        try:
            out["title"], out["skeleton"], out["skeleton_paths"] = self._driver.execute_script(_SKELETON_JS,
                                                                                              SKELETON_DEPTH)
            if coverage and self._js_coverage:
                new = 0
                for script in self._driver.execute_cdp_cmd("Profiler.takePreciseCoverage", {})["result"]:
                    url = script.get("url", "").split("?")[0]
                    if not url:
                        continue
                    for fn in script["functions"]:
                        key = zlib.crc32(f"{url}|{fn['functionName']}|{fn['ranges'][0]['startOffset']}".encode())
                        if key not in self._js_seen:
                            self._js_seen.add(key)
                            new += 1
                out["js_new"] = new
        except Exception as e:      # a page in flux or a dead browser: the step keeps what was read
            out["true_error"] = str(e).splitlines()[0][:120]
        return out

    def close(self) -> None:
        self._states.clear()
        self._saved_state_ids.clear()
        self._quit()
        with self._sandboxed():
            self._quit()
        shutil.rmtree(self._work_dir, ignore_errors=True)
