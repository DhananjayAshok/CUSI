"""
Checks the Web env's bot-check (challenge page) detector.
    python -m tests.web_challenge_test             # challenge_reason on canned pages
    python -m tests.web_challenge_test --live      # + a real browser: local challenge fixtures and the task sites
The live run fails on a false positive (a normal fixture or an included task site flagged); the excluded sites are
only reported, since whether they serve a bot check depends on the day and the network.
"""
import json
import os
import tempfile
import click
from cusi.envs.webvoyager import challenge_reason

LONG = "Real content. " * 200
CANNED = [
    # (title, body text, widget marker, is a challenge)
    ("Just a moment...", "Verifying you are human. This may take a few seconds.", True, True),
    ("Attention Required! | Cloudflare", "Sorry, you have been blocked", False, True),
    ("www.example.com", "Checking if the site connection is secure. Enable JavaScript and cookies to continue", False,
     True),
    ("Amazon.com", "Enter the characters you see below. Sorry, we just need to make sure you're not a robot.", False,
     True),
    ("Access Denied", "You don't have permission to access this server.", False, True),
    ("Robot or human?", "Activate and hold the button to confirm that you're human.", False, True),
    ("Example", "Press & Hold to confirm you are a human (and not a bot).", False, True),
    ("Sign in", "Email Password Sign in", True, True),                 # widget on a short page
    ("arXiv.org e-Print archive", LONG, False, False),
    ("Contact us", LONG + " This form is protected by reCAPTCHA.", True, False),   # widget on a long page
    ("Security research blog", LONG + " how to beat a captcha", False, False),
    ("Hugging Face", "The AI community building the future. " + LONG, False, False),
    ("Short page", "Hello world", False, False),
]

FIXTURES = {
    "cloudflare.html": ("<html><head><title>Just a moment...</title></head><body><div id='challenge-running'>"
                        "Verifying you are human. This may take a few seconds.</div></body></html>", True),
    "turnstile.html": ("<html><head><title>Example</title></head><body><p>One more step</p>"
                       "<div class='cf-turnstile'></div></body></html>", True),
    "captcha_text.html": ("<html><head><title>Example shop</title></head><body><p>Our systems have detected unusual "
                          "traffic from your computer network.</p></body></html>", True),
    "normal.html": ("<html><head><title>Recipes</title></head><body>" + "<p>A normal page about soup.</p>" * 200
                    + "<div class='g-recaptcha'></div></body></html>", False),
}


def canned() -> None:
    bad = [(t, want) for t, text, marker, want in CANNED
           if (challenge_reason(title=t, text=text, marker=marker) is not None) != want]
    for t, text, marker, want in CANNED:
        print(f"  {'ok  ' if (challenge_reason(title=t, text=text, marker=marker) is not None) == want else 'FAIL'}"
              f"  {t[:40]!r:44} -> {challenge_reason(title=t, text=text, marker=marker)}")
    if bad:
        raise SystemExit(f"FAIL: {bad}")
    print(f"canned: all {len(CANNED)} correct")


def live() -> None:
    from cusi.utils.parameter_handling import load_parameters
    from cusi.envs.webvoyager import WebVoyagerPlayEnv, excluded_sites
    parameters = load_parameters()
    fixture_dir = tempfile.mkdtemp(dir=parameters["tmp_dir"])
    failures = []
    for name, (html, want) in FIXTURES.items():
        path = os.path.join(fixture_dir, name)
        with open(path, "w") as f:
            f.write(html)
        env = WebVoyagerPlayEnv(url=f"file://{path}", mode="free_play", fast_waits=True, parameters=parameters)
        _obs, info = env.reset()
        env.close()
        got = info["challenge"]
        print(f"  {'ok  ' if got == want else 'FAIL'}  fixture {name:20} challenge={got} {info.get('challenge_reason')}")
        if got != want:
            failures.append(name)
    excluded = set(excluded_sites(parameters=parameters))
    sites = {}
    with open(os.path.join(parameters["project_root"], "WebVoyager", "data", "WebVoyager_data.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            sites.setdefault(row["web_name"], row["web"])
    for name, url in sorted(sites.items()):
        try:
            env = WebVoyagerPlayEnv(url=url, mode="free_play", fast_waits=True, parameters=parameters)
            _obs, info = env.reset()
            env.close()
        except Exception as e:
            print(f"  ERR   {name:22} {str(e)[:100]}")
            continue
        flagged = info["challenge"]
        if name in excluded:
            print(f"  info  {name:22} (excluded) challenge={flagged} {info.get('challenge_reason') or ''}")
        else:
            print(f"  {'FAIL' if flagged else 'ok  '}  {name:22} challenge={flagged} {info.get('challenge_reason') or ''}")
            if flagged:
                failures.append(name)
    if failures:
        raise SystemExit(f"FAIL: {failures}")
    print("live: no false positives")


@click.command()
@click.option("--live", "with_browser", is_flag=True)
def main(with_browser):
    canned()
    if with_browser:
        live()
    print("ALL OK")

if __name__ == "__main__":
    main()
