"""
Base-disagreement rows: which practice action rows does the base model already get right? The base model decodes
every executor action row of a dataset's success/train exactly as the eval agent would call it (one user turn, the
row's text then its images as JPEG data URIs at PIL's defaults, greedy, M3A's 1000-token budget, stop "[STOP]"), and
its action is compared with the row's after normalisation (norm_action). The rows where they differ are the ones
training can correct; build_disagreement_dataset writes them as a dataset of their own (an "arm" beside the full one).

    records = decode_base(base_url=..., model=..., dataset_dir=..., out=<jsonl>)      # resumable
    build_disagreement_dataset(src_dir=..., out_dir=..., records_path=<jsonl>, val_copy=200, seed=0)
"""
import base64
import difflib
import io
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from PIL import Image
from cusi.finetune.repro import action_of

EVAL_MAX_TOKENS = 1000     # cusi.agents.executors.m3a.M3AExecutor.MAX_NEW_TOKENS
EVAL_STOP = ["[STOP]"]     # cusi.utils.lm_inference query_client's stop list


def norm_action(text: Optional[str]) -> Optional[str]:
    """An action reply's action (the text after the last "Action:") as canonical JSON, with equivalent spellings
    merged: clear_text false = absent, a scroll's element index ignored, app names case-insensitive. None if the
    reply has no action; the raw action text if it is not JSON."""
    action = action_of(text)
    if action is None:
        return None
    try:
        parsed = json.loads(action)
    except Exception:
        return action
    if not isinstance(parsed, dict):
        return action
    if parsed.get("clear_text") is False:
        parsed.pop("clear_text")
    if parsed.get("action_type") == "scroll":
        parsed.pop("index", None)
    if isinstance(parsed.get("app_name"), str):
        parsed["app_name"] = parsed["app_name"].lower()
    return json.dumps(parsed, sort_keys=True)


def eval_messages(*, row: dict, dataset_dir: str) -> list:
    """The row's single user turn as the eval agent sends it (AgentVLM.infer): text, then JPEG images."""
    if len(row["messages"]) != 1 or row["messages"][0]["role"] != "user":
        raise ValueError(f"{row['episode_id']} call {row['call_idx']}: not a single-prompt call")
    texts = [c["text"] for c in row["messages"][0]["content"] if c["type"] == "text"]
    content = [{"type": "text", "text": "\n".join(texts)}]
    for c in row["messages"][0]["content"]:
        if c["type"] != "image":
            continue
        img = Image.open(os.path.join(dataset_dir, c["image"]))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
    return [{"role": "user", "content": content}]


def action_rows(*, dataset_dir: str) -> list:
    """(line index, row) of every executor action row of success/train with an action in its response."""
    with open(os.path.join(dataset_dir, "success", "train.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [(i, r) for i, r in enumerate(rows) if r["role"] == "executor" and r["tag"] == "action"
            and "paraphrase_of" not in r and action_of(r["response"])]


def decode_base(*, base_url: str, model: str, dataset_dir: str, out: str, workers: int = 16) -> list:
    """Every action row decoded by the served model -> records appended to out (JSONL; rows already in out are
    skipped, so a rerun resumes)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=900)
    done = {}
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["row"]] = rec
    todo = [(i, r) for i, r in action_rows(dataset_dir=dataset_dir) if i not in done]

    def one(item):
        i, row = item
        reply = client.chat.completions.create(model=model, messages=eval_messages(row=row, dataset_dir=dataset_dir),
                                               temperature=0.0, max_tokens=EVAL_MAX_TOKENS, stop=EVAL_STOP)
        text = reply.choices[0].message.content or ""
        target, base = norm_action(row["response"]), norm_action(text)
        return {"row": i, "episode_id": row["episode_id"], "task_id": row["task_id"], "call_idx": row["call_idx"],
                "target": target, "base": base, "agree": base == target, "base_raw": text}

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[rec["row"]] = rec
    return [done[i] for i in sorted(done)]


def build_disagreement_dataset(*, src_dir: str, out_dir: str, records_path: str, val_copy: int, seed: int) -> dict:
    """out_dir/success/train.jsonl = src_dir's action rows the base model disagrees with (plan and summary rows are
    left out: this arm trains only the corrections); validation = up to val_copy of them (a fit check, as
    val_copy); images/ is a link to src_dir's. -> stats."""
    with open(os.path.join(src_dir, "success", "train.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(records_path) as f:
        records = {r["row"]: r for r in (json.loads(line) for line in f if line.strip())}
    expected = {i for i, _ in action_rows(dataset_dir=src_dir)}
    if set(records) != expected:
        raise ValueError(f"{records_path} covers {len(records)} rows, {len(expected)} action rows expected")
    keep = [rows[i] for i in sorted(records) if not records[i]["agree"]]
    for split in ("success", "failure"):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    link = os.path.join(out_dir, "images")
    if not os.path.islink(link):
        os.symlink(os.path.relpath(os.path.join(src_dir, "images"), out_dir), link)
    order = sorted(random.Random(seed).sample(range(len(keep)), min(val_copy, len(keep))))
    files = {("success", "train"): keep, ("success", "validation"): [keep[i] for i in order],
             ("failure", "train"): [], ("failure", "validation"): []}
    for (kind, split), split_rows in files.items():
        with open(os.path.join(out_dir, kind, f"{split}.jsonl"), "w") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")
    stats = {"source_dataset": src_dir, "records": records_path, "action_rows": len(records),
             "base_agree": sum(r["agree"] for r in records.values()), "success_train_rows": len(keep),
             "success_validation_rows": len(order), "validation_copied_from_train": len(order),
             "tasks": len({r["task_id"] for r in keep}), "rows": "base-disagreement action rows only"}
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


# ---------------------------------------------------------------- every env, every row (opt-in: env=...)
# Each row's reply budget as the eval agent called it (greedy evals: temperature 0 for every call).
ENV_MAX_TOKENS = {
    "android": {"default": 1000},                                       # M3AExecutor.MAX_NEW_TOKENS
    "web": {"default": 1000},                                           # WebVoyagerExecutor.MAX_NEW_TOKENS
    "gameboy": {"default": 8000, "done_check": 1500,                    # GameBoyExecutor / DONE_CHECK budget
                "supervisor": 5000},                                    # SUPERVISOR_MAX_NEW_TOKENS
}


def row_max_tokens(*, env: str, row: dict) -> int:
    budgets = ENV_MAX_TOKENS[env]
    if row["role"] == "supervisor" and "supervisor" in budgets:
        return budgets["supervisor"]
    return budgets.get(row["tag"], budgets["default"])


def _data_url(*, path: str, encoding: str) -> str:
    """An image as the eval sent it: a chat call's PNG (the stored file is that PNG), a single prompt's JPEG."""
    if encoding == "png":
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    img = Image.open(path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def row_messages(*, row: dict, dataset_dir: str) -> list:
    """Any row's input as the eval agent sent it: a single prompt (AgentVLM.infer: its text, then its images as JPEG)
    or a chat (AgentVLM.chat: the turns as recorded, images as PNG in place)."""
    encoding = row.get("image_encoding", "jpeg")
    if encoding != "png":
        return eval_messages(row=row, dataset_dir=dataset_dir)
    out = []
    for m in row["messages"]:
        if isinstance(m["content"], str):
            out.append({"role": m["role"], "content": m["content"]})
            continue
        out.append({"role": m["role"], "content": [
            {"type": "text", "text": c["text"]} if c["type"] == "text" else
            {"type": "image_url", "image_url": {"url": _data_url(path=os.path.join(dataset_dir, c["image"]),
                                                                 encoding="png")}}
            for c in m["content"]]})
    return out


def sample_rows(*, rows: list, n: Optional[int], seed: int) -> list:
    """Row indices to decode: all (n None or >= len), else a seeded sample stratified by task (each task keeps its
    share, at least one row)."""
    if n is None or n >= len(rows):
        return list(range(len(rows)))
    by_task = {}
    for i, r in enumerate(rows):
        by_task.setdefault(r["task_id"], []).append(i)
    rng = random.Random(seed)
    picked = []
    for task in sorted(by_task):
        idx = by_task[task]
        k = max(1, round(n * len(idx) / len(rows)))
        picked += rng.sample(idx, min(k, len(idx)))
    return sorted(picked)


def decode_all(*, base_url: str, model: str, env: str, dataset_dir: str, out: str, workers: int = 16,
               n: Optional[int] = None, seed: int = 0) -> int:
    """Every success/train row (or a stratified sample of n) decoded by the served model as the eval agent would
    call it; raw replies appended to out (JSONL, resumable). Comparison is left to the analysis (raw kept)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=1800)
    with open(os.path.join(dataset_dir, "success", "train.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {json.loads(line)["row"] for line in f if line.strip()}
    todo = [i for i in sample_rows(rows=rows, n=n, seed=seed) if i not in done]

    def one(i):
        row = rows[i]
        try:
            reply = client.chat.completions.create(model=model, messages=row_messages(row=row, dataset_dir=dataset_dir),
                                                   temperature=0.0, max_tokens=row_max_tokens(env=env, row=row),
                                                   stop=EVAL_STOP)
            text, error = reply.choices[0].message.content or "", None
        except Exception as e:       # e.g. a row over the server's context: recorded, not fatal
            text, error = None, f"{type(e).__name__}: {str(e)[:300]}"
        return {"row": i, "env": env, "role": row["role"], "tag": row["tag"], "episode_id": row["episode_id"],
                "task_id": row["task_id"], "scene": row["scene"], "call_idx": row["call_idx"],
                "target_raw": row["response"], "base_raw": text, "error": error}

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
    return len(done) + len(todo)


# ---------------------------------------------------------------- agreement of decode_all's records
VERDICT_KEYS = ("Complete", "Flawed", "Undone")      # GameBoy judge / done_check, plan_flaw, regression_check


def action_line(text: Optional[str]) -> Optional[str]:
    """The first line after the last "Action:", whitespace collapsed (the raw action as written)."""
    if not text or "Action:" not in text:
        return None
    lines = text.rsplit("Action:", 1)[1].strip().splitlines()
    return " ".join(lines[0].split()) if lines else None


def norm_web(text: Optional[str]) -> Optional[str]:
    """A WebVoyager action, equivalent spellings merged: case-insensitive, spaces around [ ] ; ignored, a trailing
    period dropped, brackets optional around the element / WINDOW (Click [5] = click 5; Scroll [WINDOW]; down =
    scroll window;down). Typed and answered text are compared case-insensitively too."""
    a = action_line(text)
    if a is None:
        return None
    a = a.lower().rstrip(".").replace("[", " ").replace("]", " ")
    a = " ".join(a.split())
    return a.replace(" ;", ";").replace("; ", ";")


def norm_gameboy(text: Optional[str]) -> Optional[str]:
    """A GameBoy executor action: one button (UP / DOWN / LEFT / RIGHT / A / B / START), case-insensitive. The
    agent presses exactly one button per action, so order and multiplicity cannot differ."""
    a = action_line(text)
    return a.upper().strip(" .") if a else None


def verdict(text: Optional[str]) -> Optional[str]:
    """A yes / no verdict line (Complete / Flawed / Undone), the last one in the reply."""
    if not text:
        return None
    found = re.findall(r"^\s*(" + "|".join(VERDICT_KEYS) + r")\s*:\s*(yes|no)\b", text, re.MULTILINE | re.IGNORECASE)
    return f"{found[-1][0].lower()}:{found[-1][1].lower()}" if found else None


def agreement_report(*, env: str, dataset_dir: str, records_path: str) -> dict:
    """Base vs practice-model agreement per row kind: actions (exact action line and normalised), yes/no verdicts,
    free text (difflib ratio); actions also by fifth of episode, on each episode's final action, and per game /
    site (scene prefix)."""
    with open(os.path.join(dataset_dir, "success", "train.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(records_path) as f:
        recs = {r["row"]: r for r in (json.loads(line) for line in f if line.strip())}
    norm = {"web": norm_web, "gameboy": norm_gameboy, "android": norm_action}[env]
    actions_of = {}
    for i, r in enumerate(rows):
        if r["role"] == "executor" and r["tag"] == "action":
            actions_of.setdefault(r["episode_id"], []).append(i)
    pos, final = {}, set()
    for idx in actions_of.values():
        idx.sort(key=lambda i: rows[i]["call_idx"])
        final.add(idx[-1])
        for j, i in enumerate(idx):
            pos[i] = min(4, int(5 * j / len(idx)))

    def group(name):
        return name.split("/")[0] if env == "gameboy" else name

    out = {"env": env, "rows_decoded": len(recs), "rows_total": len(rows),
           "errors": sum(r["error"] is not None for r in recs.values()), "kinds": {}}
    acc = {}

    def add(key, val):
        acc.setdefault(key, []).append(val)

    for i, rec in recs.items():
        if rec["error"] is not None:
            continue
        row, kind = rows[i], f"{rows[i]['role']}/{rows[i]['tag']}"
        tgt, base = rec["target_raw"], rec["base_raw"]
        if row["role"] == "executor" and row["tag"] == "action":
            exact = action_line(tgt) is not None and action_line(tgt) == action_line(base)
            normal = norm(tgt) is not None and norm(tgt) == norm(base)
            for key in (kind, f"{kind}|fifth{pos[i]}", f"{kind}|{'final' if i in final else 'not_final'}",
                        f"{kind}|group={group(row['scene'])}"):
                add(key + "|exact", exact)
                add(key + "|norm", normal)
        elif verdict(tgt) is not None:
            add(kind + "|verdict", verdict(tgt) == verdict(base))
        else:
            add(kind + "|text_sim", difflib.SequenceMatcher(None, tgt or "", base or "").ratio())
    for key, vals in sorted(acc.items()):
        out["kinds"][key] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 4)}
    return out
