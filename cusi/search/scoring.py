"""
VLM decision scores (plans/next_plan.md §B): at selection time, each candidate node's features are checked against
what its scene's expansions have discovered. The score, the share of features still undiscovered, is P(n) in the
energy. Scores are cached per node; each decision re-scores the candidates with the highest energy.
"""
import re
from typing import Optional
from cusi.search.describe import normalise_phrase

VERDICT_VALUE = {"no": 1.0, "partly": 0.5, "yes": 0.0}

SCORE_PROMPT = """An explorer is mapping {env_description} with no particular task, trying to discover as many \
different features and functionalities as possible.

Features that could be reached by exploring from one particular screen:
{features}

Everything exploration has discovered so far:
{discovered}

For each feature above, say whether exploration has already surfaced it: yes, partly or no, with a short reason.
Reply with exactly one line per feature, in order, in this format:
<k>. <yes|partly|no> - <reason>"""

_LINE_RE = re.compile(r"^\s*(\d+)\s*[.):]\s*\**\s*(yes|partly|no)\b\**\s*[-—–:,]?\s*(.*)$", re.IGNORECASE)


def discovered_set(*, expansions) -> list:
    """The union of the expansions' `discovered` lists, normalised duplicates removed, in first-seen order."""
    out, seen = [], set()
    for e in expansions:
        for item in e.discovered or []:
            key = normalise_phrase(text=item)
            if key and key not in seen:
                seen.add(key)
                out.append(item)
    return out


def parse_verdicts(*, text: str, features: list) -> list:
    """[{"feature", "verdict", "reason"}] per feature; a feature without a parsable line gets verdict None."""
    found = {}
    for line in str(text or "").splitlines():
        m = _LINE_RE.match(line)
        if m:
            k = int(m.group(1)) - 1
            if 0 <= k < len(features) and k not in found:
                found[k] = (m.group(2).lower(), m.group(3).strip()[:300])
    return [{"feature": f, "verdict": found.get(k, (None, ""))[0], "reason": found.get(k, (None, ""))[1]}
            for k, f in enumerate(features)]


def score_of(*, verdicts: list, default_score: float) -> float:
    """Share of features still undiscovered (no = 1, partly = 0.5); unparsed verdicts count as undiscovered."""
    if not verdicts:
        return default_score
    return sum(VERDICT_VALUE.get(v["verdict"], 1.0) for v in verdicts) / len(verdicts)


class VLMScorer:
    def __init__(self, *, vlm, default_score: float = 0.5, max_new_tokens: int = 400, temperature: float = 0.0,
                 max_discovered: int = 400) -> None:
        self.vlm, self.default_score = vlm, default_score
        self.max_new_tokens, self.temperature, self.max_discovered = max_new_tokens, temperature, max_discovered
        self.n_calls = 0

    def prompt(self, *, env_description: str, features: list, discovered: list) -> str:
        shown = discovered[-self.max_discovered:]
        lines = "\n".join(f"- {d}" for d in shown) or "(nothing yet)"
        if len(discovered) > len(shown):
            lines = f"({len(discovered) - len(shown)} earlier items not shown)\n" + lines
        return SCORE_PROMPT.format(env_description=env_description, discovered=lines,
                                   features="\n".join(f"{k + 1}. {f}" for k, f in enumerate(features)))

    def score(self, *, items: list) -> list:
        """items: [{"node_id", "features", "discovered", "env_description"}] -> one entry per item,
        {"node_id", "score", "verdicts", "raw"}, from one batched text-only call (nodes without features get
        default_score and no call)."""
        asked = [it for it in items if it["features"]]
        raws = []
        if asked:
            prompts = [self.prompt(env_description=it["env_description"], features=it["features"],
                                   discovered=it["discovered"]) for it in asked]
            out = self.vlm.infer(texts=prompts, images=None, max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature)["output"]
            raws = [str(o) for o in (out if isinstance(out, list) else [out])]
            self.n_calls += len(asked)
        by_id = {it["node_id"]: raw for it, raw in zip(asked, raws)}
        entries = []
        for it in items:
            raw: Optional[str] = by_id.get(it["node_id"])
            verdicts = parse_verdicts(text=raw, features=it["features"]) if raw is not None else []
            entries.append({"node_id": it["node_id"], "score": score_of(verdicts=verdicts,
                                                                         default_score=self.default_score),
                            "verdicts": verdicts, "raw": raw})
        return entries
