"""
Temperature sweep: how hot can the eval agent's model be sampled before its replies break? Real test-set calls are
replayed from finished eval runs' leg records (the prompt or chat and its frames, exactly as the agent sent them)
through the eval client (AgentVLM, temperature fixed) at each temperature, and every reply goes through its env's
own parser: GameBoy's single-action policy, done check and supervisor plan; WebVoyager's Thought/Action split and
extract_information; M3A's reason/action parse and JSONAction. Each reply also records signs of degeneration (the
token cap reached, non-Latin text, repetition) and, for decisions, whether it picks the greedy reply's action.

One env per process (GameBoyRL and WebVoyager both have a top-level `utils` package).

    records = sweep(env="web", run_dirs=[...], model_name=..., vllm_base_url=..., temperatures=[0, 1, 2],
                    n_per_tag=40, samples=2, out=<jsonl>)      # resumable
    rows = summarise(records=records)
"""
import collections
import io
import json
import os
import random
import re
import unicodedata
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from PIL import Image

#: Tags replayed per env: (record source, tag). "leg" = an episode's legs/*.jsonl, "event" = its events.jsonl.
TAGS = {"gameboy": [("leg", "action"), ("leg", "done_check"), ("leg", "frame_diff"), ("event", "plan")],
        "web": [("leg", "action")],
        "android": [("leg", "action"), ("leg", "plan")]}
#: Reply token cap per tag: the agent's own budget, capped so a runaway reply costs at most this many tokens
#: (GameBoy's 8000 and the supervisor's 5000 are never approached by a sane reply; reaching the cap = runaway).
CAP = 1500
#: A reply is degenerate if it reaches the cap, if more than this share of its letters is outside Latin script,
#: or if it compresses below this ratio (zlib size / raw size) while being at least REPEAT_MIN_CHARS long.
NON_LATIN_MAX = 0.02
REPEAT_RATIO_MIN = 0.12
REPEAT_MIN_CHARS = 400


def _budget(*, env: str, tag: str) -> int:
    from cusi.agents.supervisors.base import SUPERVISOR_MAX_NEW_TOKENS
    if env == "gameboy":
        from cusi.agents.executors.gameboy import DONE_CHECK_MAX_NEW_TOKENS, GameBoyExecutor
        budget = {"done_check": DONE_CHECK_MAX_NEW_TOKENS, "plan": SUPERVISOR_MAX_NEW_TOKENS}.get(
            tag, GameBoyExecutor.MAX_NEW_TOKENS)
    elif env == "web":
        from cusi.agents.executors.webvoyager import WebVoyagerExecutor
        budget = WebVoyagerExecutor.MAX_NEW_TOKENS
    else:
        from cusi.agents.executors.m3a import M3AExecutor
        budget = M3AExecutor.MAX_NEW_TOKENS
    return min(budget, CAP)


def load_calls(*, env: str, run_dirs: list, n_per_tag: int, seed: int) -> list:
    """A seeded sample of n_per_tag recorded calls per tag, spread over episodes, each with its episode dir."""
    pools = collections.defaultdict(list)
    for run_dir in run_dirs:
        root = os.path.join(run_dir, "episodes")
        for episode in sorted(os.listdir(root)):
            ep = os.path.join(root, episode)
            sources = {"event": [os.path.join(ep, "events.jsonl")],
                       "leg": [os.path.join(ep, "legs", f) for f in sorted(os.listdir(os.path.join(ep, "legs")))]
                       if os.path.isdir(os.path.join(ep, "legs")) else []}
            for source, tag in TAGS[env]:
                for path in sources[source]:
                    if not os.path.exists(path):
                        continue
                    for i, line in enumerate(open(path)):
                        rec = json.loads(line)
                        rec_tag = rec.get("stage") if source == "event" else rec.get("tag")
                        if rec_tag != tag or (source == "event" and rec.get("kind") != "supervisor"):
                            continue
                        if not (rec.get("prompt") or rec.get("messages")):
                            continue
                        pools[tag].append({"id": f"{os.path.basename(run_dir)}/{episode}/"
                                                 f"{os.path.basename(path)}:{i}", "tag": tag, "episode_dir": ep,
                                           "record": rec})
    rng = random.Random(seed)
    out = []
    for tag, pool in sorted(pools.items()):
        # one call per episode first, so the sample spans tasks, then fill from the rest
        by_ep = collections.defaultdict(list)
        for c in pool:
            by_ep[c["episode_dir"]].append(c)
        firsts = [rng.choice(v) for _, v in sorted(by_ep.items())]
        rng.shuffle(firsts)
        rest = [c for c in pool if c not in firsts]
        rng.shuffle(rest)
        out += (firsts + rest)[:n_per_tag]
    return out


def _images(*, episode_dir: str, refs: list) -> list:
    out = []
    for ref in refs:
        archive, name = ref.split("/", 1)
        with zipfile.ZipFile(os.path.join(episode_dir, archive)) as z:
            out.append(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
    return out


class Parser:
    """The env's own validity test of a reply, and the decision's key (for agreement with the greedy reply)."""

    def __init__(self, *, env: str, calls: list) -> None:
        self.env = env
        if env == "gameboy":
            from cusi.agents.executors.gameboy import ACTION_POLICY, parse_completion
            from cusi.utils.parsing import parse_plan
            self._policy, self._completion, self._plan = ACTION_POLICY(), parse_completion, parse_plan
            # the controller's action names: every action a recorded step executed
            self._actions = {s["action"] for c in calls if c["tag"] == "action"
                             for s in c["record"].get("steps", []) if s.get("valid")}
        elif env == "web":
            from cusi.envs.webvoyager import load_webvoyager
            from cusi.utils import load_parameters
            self._wv = load_webvoyager(project_root=load_parameters()["project_root"])
        else:
            from android_world.agents import agent_utils, m3a_utils
            from android_world.env import json_action
            from cusi.agents.executors.m3a import parse_subgoals
            from cusi.finetune.disagree import norm_action
            self._m3a, self._agent, self._json_action = m3a_utils, agent_utils, json_action
            self._subgoals, self._norm_action = parse_subgoals, norm_action

    def check(self, *, tag: str, response: str) -> tuple:
        """(valid, key, why-invalid)."""
        if self.env == "gameboy":
            if tag == "action":
                decision = self._policy.parse(response)
                if decision is None:
                    return False, None, "parse failure"
                action = decision.actions[0]
                return (action in self._actions), action, None if action in self._actions else "unknown action"
            if tag == "done_check":
                verdict = self._completion(response)
                return verdict is not None, verdict, None if verdict is not None else "no Complete line"
            if tag == "plan":
                steps = self._plan(response)
                return bool(steps), len(steps), None if steps else "no plan"
            return bool(response.strip()), None, None if response.strip() else "empty"
        if self.env == "web":
            if "Thought:" not in response or "Action:" not in response:
                return False, None, "format error"
            chosen = re.split(r"Thought:|Action:|Observation:", response)[2].strip()
            if chosen.startswith("Google"):
                return False, None, "pruned action"
            key, _ = self._wv.extract_information(chosen)
            return key is not None, " ".join(chosen.split()) if key else None, None if key else "unknown action"
        if tag == "plan":
            subgoals = self._subgoals(response=response)
            return bool(subgoals), len(subgoals), None if subgoals else "no subgoals"
        try:
            reason, action = self._m3a.parse_reason_action_output(response)
        except TypeError:
            return False, None, "unparseable action"
        if not reason or not action:
            return False, None, "bad format"
        parsed = self._agent.extract_json(action)
        try:
            self._json_action.JSONAction(**parsed)
        except Exception:
            return False, None, "unparseable action"
        return True, self._norm_action(response), None


def degeneration(*, response: str, output_tokens: Optional[int], cap: int) -> dict:
    letters = [ch for ch in response if ch.isalpha()]
    non_latin = sum(1 for ch in letters if "LATIN" not in unicodedata.name(ch, "")) / max(1, len(letters))
    raw = response.encode("utf-8")
    ratio = len(zlib.compress(raw)) / max(1, len(raw))
    return {"hit_cap": output_tokens is not None and output_tokens >= cap - 1, "non_latin": round(non_latin, 4),
            "repetitive": len(raw) >= REPEAT_MIN_CHARS and ratio < REPEAT_RATIO_MIN}


def sweep(*, env: str, run_dirs: list, model_name: str, vllm_base_url: str, temperatures: list, n_per_tag: int,
          samples: int, out: str, seed: int = 0, workers: int = 32) -> list:
    """Every sampled call at every temperature, `samples` replies each, appended to `out` (resumes by key)."""
    from cusi.agents.vlm import AgentVLM
    calls = load_calls(env=env, run_dirs=run_dirs, n_per_tag=n_per_tag, seed=seed)
    parser = Parser(env=env, calls=calls)
    clients = {t: AgentVLM(model_name=model_name, vllm_base_url=vllm_base_url, temperature=float(t))
               for t in temperatures}
    done = set()
    if os.path.exists(out):
        done = {(r["id"], r["temperature"], r["sample"]) for r in map(json.loads, open(out))}
    jobs = [(c, t, s) for c in calls for t in temperatures for s in range(samples)
            if (c["id"], float(t), s) not in done]

    def run(call, t, s):
        rec, cap = call["record"], _budget(env=env, tag=call["tag"])
        images = _images(episode_dir=call["episode_dir"], refs=rec.get("images") or [])
        if rec.get("messages"):
            res = clients[t].chat(messages=rec["messages"], images=images, max_new_tokens=cap)
        else:
            res = clients[t].infer(texts=rec["prompt"], images=images or None, max_new_tokens=cap)
        response = res["output"] if isinstance(res["output"], str) else res["output"][0]
        meta = res.get("meta") or {}
        tokens = meta.get("output_tokens")
        tokens = tokens[0] if isinstance(tokens, list) else tokens
        valid, key, why = parser.check(tag=call["tag"], response=response)
        return {"env": env, "id": call["id"], "tag": call["tag"], "temperature": float(t), "sample": s,
                "valid": valid, "key": None if key is None else str(key), "why": why, "output_tokens": tokens,
                **degeneration(response=response, output_tokens=tokens, cap=cap), "response": response}

    with open(out, "a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, *j) for j in jobs]
        for i, future in enumerate(as_completed(futures)):
            try:
                row = future.result()
            except Exception as e:
                print(f"[sweep] call failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 200 == 0:
                print(f"[sweep] {env}: {i + 1}/{len(jobs)} replies", flush=True)
    return [json.loads(line) for line in open(out)]


def summarise(*, records: list) -> list:
    """Per (env, tag, temperature): valid rate, degeneration rates, mean tokens, agreement with greedy (T=0)."""
    greedy = {}
    for r in records:
        if r["temperature"] == 0 and r["key"] is not None:
            greedy.setdefault(r["id"], r["key"])
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["env"], r["tag"], r["temperature"])].append(r)
    rows = []
    for (env, tag, t), rs in sorted(groups.items()):
        agree = [r["key"] == greedy[r["id"]] for r in rs if r["id"] in greedy]
        toks = [r["output_tokens"] for r in rs if r["output_tokens"] is not None]
        rows.append({"env": env, "tag": tag, "temperature": t, "n": len(rs),
                     "valid": round(sum(r["valid"] for r in rs) / len(rs), 3),
                     "hit_cap": round(sum(r["hit_cap"] for r in rs) / len(rs), 3),
                     "non_latin": round(sum(r["non_latin"] > NON_LATIN_MAX for r in rs) / len(rs), 3),
                     "repetitive": round(sum(r["repetitive"] for r in rs) / len(rs), 3),
                     "agree_greedy": round(sum(agree) / len(agree), 3) if agree else None,
                     "mean_tokens": round(sum(toks) / len(toks), 1) if toks else None,
                     "invalid_why": dict(collections.Counter(r["why"] for r in rs if not r["valid"]))})
    return rows
