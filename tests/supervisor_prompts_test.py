"""
Checks the supervisor prompts: GameBoy's equal GameBoyRL's; Android/web keep placeholders and drop game wording.
    python tests/supervisor_prompts_test.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GameBoyRL"))

from cusi.agents.supervisors import prompts                      # noqa: E402
from execution.supervisors import prompts as gb_prompts   # noqa: E402

# Placeholders the supervisor code fills, per prompt.
FILLED = {
    "PLAN_PROMPT": {"[TASK]", "[INSIGHTS]"},
    "FILTER_INSIGHTS_PROMPT": {"[TASK]", "[CANDIDATES]"},
    "DISTILL_INSIGHTS_PROMPT": {"[TASK]", "[CANDIDATES]"},
    "JUDGE_SLICE_PROMPT": {"[TASK]", "[START_IDX]", "[END_IDX]", "[TOTAL]", "[ACTION_SEQUENCE]"},
    "JUDGE_CONSOLIDATE_PROMPT": {"[TASK]", "[SEGMENT_SUMMARIES]", "[STOP_REASON]"},
    "REGRESSION_CHECK_PROMPT": {"[PREVIOUS_STEP]", "[EARLIER_STEPS_BLOCK]", "[SEGMENT_SUMMARIES]"},
    "RESUME_HINT_PROMPT": {"[TASK]", "[SEGMENT_SUMMARIES]", "[JUDGEMENT]", "[REGRESSION_BLOCK]", "[INSIGHTS]",
                           "[PRIOR_HINT_BLOCK]"},
    "PLAN_FLAW_PROMPT": {"[OVERALL_TASK]", "[PLAN_BLOCK]", "[FAILURE_HISTORY]", "[JUDGEMENT]", "[REGRESSION_LINE]",
                         "[INSIGHTS]"},
    "RELEVANCE_PROMPT": {"[TASK]", "[KIND]", "[ENTRY]", "[FRAME_NOTE]", "[EVIDENCE_NOTE]"},
}
LITERAL = {"[STEP]", "[STOP]"}
# ("button" is allowed: Android and web pages have on-screen buttons.)
GAME_WORDS = re.compile(r"\b(game|games|player|players|playthrough|playthroughs|press|pressed)\b", re.IGNORECASE)


def main() -> None:
    gb = prompts.prompts_for(domain=prompts.GAMEBOY)
    for name in prompts.PROMPT_NAMES:
        assert gb[name] == getattr(gb_prompts, name), f"GameBoy {name} differs from GameBoyRL's"
    print(f"GameBoy: {len(gb)} prompts equal GameBoyRL's")

    for env in ("android", "web"):
        texts = prompts.prompts_for(domain=prompts.DOMAINS[env])
        for name, text in texts.items():
            found = set(re.findall(r"\[[A-Z_]+\]", text)) - LITERAL
            assert found == FILLED[name], f"{env} {name}: placeholders {sorted(found)} != {sorted(FILLED[name])}"
            words = GAME_WORDS.findall(text)
            assert not words, f"{env} {name}: game wording left: {sorted(set(words))}"
        domain = prompts.DOMAINS[env]
        for field in ("trace_heading", "trace_block_heading", "regression_tail", "relevance_frame_note",
                      "relevance_no_frame_note"):
            assert not GAME_WORDS.findall(getattr(domain, field)), f"{env} {field} has game wording"
        print(f"{env}: {len(texts)} prompts, placeholders intact, no game wording")
    print("PROMPT TEST OK")


if __name__ == "__main__":
    main()
