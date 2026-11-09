"""
Does a practice adapter reproduce its training data? For executor action rows of the adapter's own practice dataset
(success/train, originals only, and success/validation: held-out practice tasks), the base model and the adapter
each decode the row's input greedily on one vLLM server (the adapter as a LoRA module), and their action is compared
with the row's (the text after the last "Action:", first line, whitespace collapsed).

If the adapter matches the training action no more often than the base, training did not move the decisions it was
trained on; if it matches far more often while the test eval stays flat, training strength is not the limit.
"""
import base64
import io
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from cusi.finetune.data import chat_messages, read_rows


def action_of(text: str):
    if not text or "Action:" not in text:
        return None
    line = text.rsplit("Action:", 1)[1].strip().splitlines()
    return re.sub(r"\s+", " ", line[0]).strip() if line else None


def to_openai(*, row: dict) -> list:
    """The row's input as OpenAI chat messages, images as PNG data URIs in their original places."""
    messages, images = chat_messages(row=row)
    it = iter(images)
    out = []
    for m in messages:
        parts = []
        for p in m["content"]:
            if p["type"] == "image":
                buf = io.BytesIO()
                next(it).save(buf, format="PNG")
                parts.append({"type": "image_url",
                              "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
            else:
                parts.append({"type": "text", "text": p["text"]})
        out.append({"role": m["role"], "content": parts})
    return out


def sample_rows(*, dataset_dir: str, split: str, n: int, seed: int) -> list:
    rows = read_rows(path=os.path.join(dataset_dir, "success", f"{split}.jsonl"), roles=("executor",),
                     dataset_dir=dataset_dir)
    rows = [r for r in rows if r["tag"] == "action" and "paraphrase_of" not in r and action_of(r["response"])]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def run(*, base_url: str, base_model: str, lora_model: str, dataset_dir: str, n: int, out: str, workers: int,
        max_tokens: int, seed: int = 0) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    def decode(model: str, messages: list) -> str:
        r = client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
                                           stop=["[STOP]"])
        return r.choices[0].message.content or ""

    summary = {"dataset_dir": dataset_dir, "lora_model": lora_model, "n_per_split": n}
    records = []
    for split in ("train", "validation"):
        rows = sample_rows(dataset_dir=dataset_dir, split=split, n=n, seed=seed)

        def one(row):
            msgs = to_openai(row=row)
            b, a = decode(base_model, msgs), decode(lora_model, msgs)
            target = action_of(row["response"])
            return {"split": split, "task_id": row["task_id"], "target": target, "base": action_of(b),
                    "adapter": action_of(a), "base_raw_tail": b[-200:], "adapter_raw_tail": a[-200:]}

        with ThreadPoolExecutor(workers) as ex:
            recs = list(ex.map(one, rows))
        records += recs
        k = len(recs)
        summary[split] = {
            "n": k,
            "base_match": sum(r["base"] == r["target"] for r in recs) / max(1, k),
            "adapter_match": sum(r["adapter"] == r["target"] for r in recs) / max(1, k),
            "base_adapter_agree": sum(r["base"] == r["adapter"] for r in recs) / max(1, k),
            "base_unparsed": sum(r["base"] is None for r in recs) / max(1, k),
            "adapter_unparsed": sum(r["adapter"] is None for r in recs) / max(1, k)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    return summary
