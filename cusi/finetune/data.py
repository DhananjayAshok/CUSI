"""
The practice success rows (cusi.practice.stages.dataset) as training examples: one row is one model call, supervisor
or executor, whose messages are the input and response the reply to learn. The sequence is exactly what the served
model sees and generates: the chat template's generation prompt (Gemma 4's ends in an empty thought channel), then the
response and the end-of-turn token; only the response and end-of-turn tokens carry loss.

Images are re-encoded as the call sent them (as_sent: JPEG for single-prompt calls, PNG for chat calls), so training
sees the pixels the served model saw. Rows are batched by the processor itself (right padding keeps every row's tokens
as encoded alone; the images of a micro-batch are concatenated in row order). A length index (every row's prompt and
full token counts, computed once and cached) drops over-long rows and packs rows of similar length into micro-batches
within a padded-token budget.
"""
import io
import json
import os
import torch
from PIL import Image

IGNORE_INDEX = -100
END_OF_TURN = "<turn|>"


def read_rows(*, path: str, roles: tuple) -> list:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["role"] in roles]


def dataset_paths(*, dataset_dir: str) -> dict:
    paths = {split: os.path.join(dataset_dir, "success", f"{split}.jsonl") for split in ("train", "validation")}
    for split, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"no {split} success rows at {path}: run the practice pipeline's dataset stage")
    return paths


def as_sent(*, image: Image.Image, encoding: str) -> Image.Image:
    """The image as the served model receives it. Single-prompt calls (AgentVLM.infer) send JPEG at PIL's defaults,
    after converting RGBA / P / LA to RGB (cusi.utils.lm_inference get_encoded_images); chat calls (AgentVLM.chat)
    send lossless PNG."""
    if encoding == "png":
        return image.convert("RGB")
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def chat_messages(*, row: dict) -> tuple:
    """row["messages"] with its image paths opened as the model received them -> (messages, images in order). Rows
    without image_encoding predate it and came from single-prompt calls (JPEG): only the Web executor's action calls
    are chat calls."""
    encoding = row.get("image_encoding", "jpeg")
    messages, images = [], []
    for m in row["messages"]:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        out = []
        for c in content:
            if c["type"] == "image":
                images.append(as_sent(image=Image.open(c["image"]), encoding=encoding))
                out.append({"type": "image"})
            else:
                out.append(c)
        messages.append({"role": m["role"], "content": out})
    return messages, images


class RowEncoder:
    """Rows -> the model's inputs (one row, or a right-padded micro-batch) with labels on the response tokens."""

    def __init__(self, *, processor) -> None:
        self.processor = processor
        if END_OF_TURN not in processor.tokenizer.get_vocab():
            raise ValueError(f"{END_OF_TURN!r} is not a token of this processor: set the end-of-turn marker for this "
                             "model family")

    def texts(self, *, row: dict) -> tuple:
        """-> (generation prompt, prompt + response + end of turn, images)."""
        messages, images = chat_messages(row=row)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt, prompt + row["response"] + END_OF_TURN, images

    def lengths(self, *, row: dict) -> tuple:
        """-> (prompt tokens, full tokens)."""
        prompt, full, images = self.texts(row=row)
        kwargs = {"images": images} if images else {}
        n_prompt = self.processor(text=prompt, return_tensors="pt", **kwargs)["input_ids"].shape[1]
        n_full = self.processor(text=full, return_tensors="pt", **kwargs)["input_ids"].shape[1]
        return int(n_prompt), int(n_full)

    def batch(self, *, rows: list, prompt_lens: list) -> dict:
        """A right-padded micro-batch; labels mask each row's prompt (prompt_lens, from the length index) and padding."""
        texts, images = [], []
        for row in rows:
            _, full, imgs = self.texts(row=row)
            texts.append(full)
            images.append(imgs)
        # one image list per row (empty for a text-only row): the processor matches them to the texts
        kwargs = {"images": images} if any(images) else {}
        self.processor.tokenizer.padding_side = "right"
        enc = self.processor(text=texts, padding=True, return_tensors="pt", **kwargs)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = IGNORE_INDEX
        for i, n in enumerate(prompt_lens):
            labels[i, :n] = IGNORE_INDEX
        enc["labels"] = labels
        return dict(enc)


def length_index(*, rows: list, encoder: RowEncoder, path: str, rank: int, world: int, gather) -> list:
    """Every row's (prompt tokens, full tokens), cached in path; ranks split the work and gather (gather: a list of
    this rank's [(row index, lengths)] -> every rank's lists)."""
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        if len(cached) == len(rows):
            return [tuple(x) for x in cached]
    mine = [(i, encoder.lengths(row=rows[i])) for i in range(rank, len(rows), world)]
    out: list = [None] * len(rows)
    for part in gather(mine):
        for i, lens in part:
            out[i] = tuple(lens)
    if rank == 0:
        with open(path + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
    return out


def pack(*, rows: list, lengths: list, max_batch_tokens: int, world: int) -> list:
    """rows, sorted by length, packed greedily into micro-batches whose padded size (rows x longest) stays within
    max_batch_tokens; then made a multiple of world (FSDP: every rank runs as many passes) by halving the micro-batches
    with the most rows, and as a last resort by zero-weight copies of a one-row micro-batch.
    -> [(row indices, weight)]."""
    micro, cur = [], []
    for i in sorted(rows, key=lambda i: lengths[i][1]):
        if cur and (len(cur) + 1) * lengths[i][1] > max_batch_tokens:
            micro.append(cur)
            cur = []
        cur.append(i)
    if cur:
        micro.append(cur)
    while len(micro) % world:
        k = max(range(len(micro)), key=lambda j: len(micro[j]))
        if len(micro[k]) < 2:
            break
        m = micro.pop(k)
        micro[k:k] = [m[:len(m) // 2], m[len(m) // 2:]]
    entries = [(m, 1.0) for m in micro]
    while len(entries) % world:
        entries.append((micro[0][:1], 0.0))
    return entries


def step_plan(*, order: list, lengths: list, rows_per_step: int, max_batch_tokens: int, world: int) -> list:
    """The row order cut into optimizer steps of rows_per_step rows (a short last step is dropped), each packed into
    micro-batches (pack) dealt round-robin to the ranks. -> [step: [rank: [(row indices, weight)]]]."""
    plan = []
    for start in range(0, len(order) - rows_per_step + 1, rows_per_step):
        entries = pack(rows=order[start:start + rows_per_step], lengths=lengths, max_batch_tokens=max_batch_tokens,
                       world=world)
        plan.append([entries[r::world] for r in range(world)])
    return plan


def row_order(*, n: int, epochs: int, seed: int) -> list:
    """Every row index once per epoch, each epoch shuffled with its own seed."""
    order = []
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(seed + epoch)
        order += torch.randperm(n, generator=g).tolist()
    return order


class MicroBatches(torch.utils.data.Dataset):
    """This rank's micro-batches ([(row indices, weight)]), in order, encoded in DataLoader workers: item -> (inputs,
    label tokens, weight)."""

    def __init__(self, *, rows: list, lengths: list, micro_batches: list, processor) -> None:
        self.rows, self.lengths, self.micro_batches = rows, lengths, micro_batches
        self.encoder = RowEncoder(processor=processor)

    def __len__(self) -> int:
        return len(self.micro_batches)

    def __getitem__(self, k: int) -> tuple:
        idx, weight = self.micro_batches[k]
        batch = self.encoder.batch(rows=[self.rows[i] for i in idx], prompt_lens=[self.lengths[i][0] for i in idx])
        return batch, int((batch["labels"] != IGNORE_INDEX).sum()), weight


def label_tokens(*, lengths: list, idx: list) -> int:
    """The loss-bearing (response + end-of-turn) tokens of rows idx."""
    return sum(lengths[i][1] - lengths[i][0] for i in idx)


def keep_rows(*, lengths: list, max_length: int) -> list:
    """The indices of the rows of at most max_length tokens."""
    return [i for i, (_, n) in enumerate(lengths) if n <= max_length]
