# WebVoyager plan

Plan for the web environment (`cusi_envs/webvoyager.py`, the `WebVoyager` submodule) across
practice, curiosity and test. Sections are added as they come up.

---

## 1. Challenge pages and site exclusion

### The problem

Some sites sit behind a bot-protection service (Cloudflare, Google reCAPTCHA, Amazon's own
check). When it suspects automation, it serves an interstitial **instead of** the site:

| service | what the page looks like |
|---|---|
| Cloudflare | "Just a moment...", "Verify you are human", "Performing security verification" (Turnstile) |
| Google | `google.com/sorry/...` "Our systems have detected unusual traffic", reCAPTCHA image grid |
| Amazon | "Continue shopping" / "Enter the characters you see below" interstitial |

A real user clears these with a check or a click. Our headless, automated browser on a
university IP usually cannot:
- Cloudflare's challenge never clears headless.
- WebVoyager's labeller emits a reCAPTCHA widget as one element, so its tiles can't be clicked.

The episode is then about the challenge page, not the site:
- **Test:** the task is unwinnable, and that is an access failure, not a model failure.
- **Practice / curiosity:** proposals, attempts and training data end up being about the
  challenge page. SynthAgent's first task-generation run on Cambridge Dictionary produced
  exactly this.

What triggers a challenge is mostly outside our control: IP reputation, request rate, and
how Chrome connects. Two browser-side signals can be changed:
- **`navigator.webdriver`.** The WebDriver standard makes an automated browser report `true`.
  `--disable-blink-features=AutomationControlled` turns off the Chrome switch that sets it.
- **User agent.** WebVoyager's headless user-agent string claims Chrome 119. The container's
  Chromium is 154, so the version headers and the string disagree, which is a bot signal.

Both only lower the block rate; neither prevents blocks. They also make the browser differ
from a plain WebVoyager run.

### Decision

**Sites with a high block rate are excluded from both train and test.**
- Train covers practice and curiosity start pages.
- Test covers the WebVoyager task set.
- Results are reported as "WebVoyager without sites X, Y, Z", with the list stated every time.

We don't try to beat bot protection on those sites.

### What we already know (SetupAttempt, gemma-4-31b-it, same cluster)

| site | tasks | evidence | category |
|---|---|---|---|
| **Amazon** | 41 | 0/10 on the shard-0 full run. Bot interstitial, the agent clicks "Continue" forever, falls back to Google, gets reCAPTCHA | blocked |
| **Google Search** | 43 | 1/11. reCAPTCHA image grid, unclickable (one element) | blocked |
| **Cambridge Dictionary** | 43 | 0/11. First page loads, every later navigation in the session gets Cloudflare Turnstile (reproduced with plain Playwright) | blocked |
| **Allrecipes** | 45 | 2/12. Cloudflare "Verify you are human" for headless Chrome even with a normal user agent (SynthAgent, WebCoach). In the full run it showed up as tab crashes | blocked |
| Booking, Google Flights, Google Map | 44, 42, 41 | Flagged as "expected to CAPTCHA" but scored 4/11, 3/11, 9/10. The low scores look like calendar/filter difficulty, not blocking. Unverified | probe |
| BBC News | 42 | 0/11, but not a block: the cookie modal's "Accept and Continue" can't be clicked. A WebVoyager element-mapping bug | **fixable, not excluded** |
| ArXiv, GitHub, Huggingface, Apple, Coursera, ESPN, Wolfram Alpha | | No blocking seen. ArXiv, GitHub and Huggingface survived repeated navigation in SynthAgent | keep |

Per-site counts are n≈11 from one shard, so they're weak as success rates. The blocking
**mechanisms** were read from agent logs and are what matter here.

**Initial exclusion list: Amazon, Google Search, Cambridge Dictionary, Allrecipes**
(172 of 643 tasks). This is applied now, before the probe, since all four have a
mechanism-level reason. The probe can add sites, and can remove one only with strong evidence.

### Plan

1. **Exclusion list in config.** Add `configs/web_vars.yaml` with
   `web_excluded_sites: [Amazon, Google Search, Cambridge Dictionary, Allrecipes]`, using
   WebVoyager's `web_name` values. One helper (e.g. `cusi_envs/webvoyager.py`:
   `allowed_sites(parameters)`) is the only place that reads it.
   - **Test:** WebVoyager tasks are filtered by `web_name` before any run. A filtered task
     file (`WebVoyager_data.cusi.jsonl`, written from the list, never edited by hand) is what
     eval runs on. Its task count is printed with every result.
   - **Train:** the practice web scenes (`cusi_practice/envs.py`, currently arXiv + GitHub)
     and curiosity's start pages come only from allowed sites. Constructing a scene on an
     excluded site is an error, not a warning.
   - **Domains, not just names.** Google Search shares `google.com` with Google Flights and
     Google Map. Exclusion is by `web_name`/start URL, while the detector (step 3) catches
     `google.com/sorry` wherever it appears.
2. **Probe job** (`slurm/web_block_probe.sh` + `scripts/slurm/web_block_probe.sh`, no GPU,
   runs in the container):
   - For each of the 15 sites, run N fresh browser sessions (e.g. 5). Each session opens the
     start page, then follows a few random in-site links and scrolls (about 5 navigations).
     Cambridge only blocks from the second navigation on, so one load is not enough.
   - Record, per site: the fraction of sessions that hit a challenge page, and on which
     navigation.
   - Run it twice: with the current WebVoyager browser settings, and with the two tweaks
     above (real-version user agent + `AutomationControlled` off). Results go to
     `results/web_block_probe_<job>.md`.
   - **Exclusion rule:** a site is excluded if it hits a challenge in **≥ 1/3 of sessions**
     under the settings we run with. The probe is rerun occasionally (sites change their
     protection), and the list in config is updated from it, with the date.
3. **Challenge detector in the env** (`cusi_envs/webvoyager.py`): `is_challenge_page()` checks
   the page title, URL and visible text against the patterns in the table above. After each
   page open and each step:
   - if a challenge shows, wait up to ~25 s for it to clear (some are transient checks);
   - if it's still there, set `info["challenge"] = True` (plus the pattern that matched).

   This is the safety net for allowed sites that block now and then: Google `/sorry` after
   many searches, or the agent leaving the site through the `Google` action.
4. **Use of the detector:**
   - **Practice:** propose skips a scene whose start page is a challenge. Attempt and practice
     runs that hit one are marked. Clean drops those calls, and dataset never includes a step
     taken on a challenge page.
   - **Curiosity:** challenge frames are not stored in the novelty buffer or replay. A
     challenge page would look maximally novel and attract the policy.
   - **Test:** a task that hits a challenge still counts as a failure (no silent exclusion
     after the fact). It's logged with the URL, and each report gives the challenge-hit count
     per site, so a site drifting into blocking shows up and goes back to step 2.
5. **Browser tweaks (user agent, `AutomationControlled`):** decided from the probe's two
   settings. If they don't change which sites pass the rule, leave them off and keep the
   browser identical to WebVoyager's. If we adopt them, they go behind a `driver_config` flag
   in the `WebVoyager` submodule (`cusi` branch), on for CUSI runs and recorded in each
   result.
6. **BBC News** is not a block. Fix the cookie-modal click separately (later section), and
   keep the site.

### Open questions

- Whether the `Google` action stays in practice/curiosity. It sends the agent to Google
  Search, an excluded site. Removing it in free play was already suggested in SetupAttempt.
  Test keeps it, since it's part of the benchmark.
- Whether 1/3 is the right threshold. Set it after seeing the probe's distribution. Most
  sites should be near 0 or near 1.
