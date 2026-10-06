"""Region-novelty stores kept inside a NoveltyArchive (archive.regions), used by the `region` scorer.

    TextLineBuffer   Android / Web (curiosity_plan §4, option A): the set of seen element lines
                     (cusi.state.text); score = share of the screen's lines never seen. Above
                     max_size, a random half is kept. (Was cusi.explore's TextNoveltyBuffer.)
                     TODO (OCR channel for Android/Web): option A may have to change to element
                     crops (B) or real OCR (C), e.g. if tree/DOM text misses text drawn into
                     images or canvases, or if a channel uniform with GameBoy is wanted.
    OCRRegionBuffer  GameBoy: GameBoyRL's OCRBuffer, ported unchanged. Not text recognition:
                     GameBoyWorlds returns pixel crops of fixed screen regions
                     (info["text_regions"]["ocr_regions"], e.g. dialogue / menu). Each crop is cut
                     into 8 vertical strips; a strip is novel if no stored strip of that region is
                     within 0.001 per pixel; region score = share of novel strips (1.0 the first time
                     a region appears); reward = max over regions. Random subsampling at 10,000.

Both: score(...) scores and adds; copy(); save(directory) / load(directory).
"""
import copy
import os
from typing import Optional
import numpy as np
import torch
from cusi.state.text import element_lines

TEXT_LINES = "text_lines"
OCR = "ocr"
TEXT_FILE = "text_buffer.txt"


class TextLineBuffer:
    def __init__(self, *, max_size: int = 50_000, seed: int = 42) -> None:
        self.max_size = max_size
        self._rng = np.random.default_rng(seed)
        self.seen: set = set()

    def score(self, *, texts: dict, add: bool = True) -> float:
        lines = element_lines(texts=texts or {})
        if not lines:
            return 0.0
        new = [l for l in lines if l not in self.seen]
        if add:
            self.seen.update(new)
            if len(self.seen) > self.max_size:
                keep = self._rng.choice(len(self.seen), self.max_size // 2, replace=False)
                items = list(self.seen)
                self.seen = {items[i] for i in keep}
        return len(new) / len(lines)

    def copy(self) -> "TextLineBuffer":
        return copy.deepcopy(self)

    def save(self, *, directory: str, merge: bool = False) -> None:
        if not self.seen:
            return
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, TEXT_FILE)
        merged = set(self.seen)
        if merge and os.path.exists(path):
            with open(path) as f:
                merged |= set(line.rstrip("\n") for line in f if line.strip())
        with open(path, "w") as f:
            f.write("\n".join(sorted(merged)) + "\n")

    @classmethod
    def load(cls, *, directory: str) -> Optional["TextLineBuffer"]:
        path = os.path.join(directory, TEXT_FILE)
        if not os.path.exists(path):
            return None
        buf = cls()
        with open(path) as f:
            buf.seen = set(line.rstrip("\n") for line in f if line.strip())
        return buf


def chunk_ocr_frame(frame: np.ndarray, n_chunks: int = 8) -> list:
    """n_chunks equal-width vertical strips of a (height, width) crop (GameBoyRL)."""
    _, width = frame.shape
    if width < n_chunks:
        return [frame]
    w = width // n_chunks
    return [frame[:, i * w:(i + 1) * w] for i in range(n_chunks)]


class OCRRegionBuffer:
    def __init__(self, *, n_chunks: int = 8, max_size: int = 10_000) -> None:
        self.n_chunks, self.max_size = n_chunks, max_size
        self.buffers: dict = {}          # region -> (n, strip_size) float32

    def _frames_to_chunks(self, frames: np.ndarray) -> torch.Tensor:
        chunks = [c.astype(np.float32) / 255.0 for frame in frames for c in chunk_ocr_frame(frame, self.n_chunks)]
        return torch.tensor(np.stack([c.reshape(-1) for c in chunks]), dtype=torch.float32)

    @staticmethod
    def unseen(chunks: torch.Tensor, buffer: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if buffer is None:
            return chunks
        max_diff = (chunks.unsqueeze(1) - buffer.unsqueeze(0)).abs().amax(dim=-1)   # (n, buffer)
        keep = max_diff.amin(dim=1) >= 0.001
        return chunks[keep] if bool(keep.any()) else None

    def _rationalize(self, key: str) -> None:
        generator = torch.Generator().manual_seed(42)
        keep = torch.randperm(self.buffers[key].shape[0], generator=generator)[:self.max_size // 2]
        self.buffers[key] = self.buffers[key][keep]

    def score(self, *, text_regions: Optional[dict], add: bool = True) -> float:
        if not text_regions or "ocr_regions" not in text_regions:
            return 0.0            # no OCR screen up
        scores, to_add = [], {}
        for region, value in text_regions["ocr_regions"].items():
            if region.startswith("_"):
                continue
            frames = np.asarray(value[0])
            frames = frames.reshape(frames.shape[0], frames.shape[1], frames.shape[2])   # n_frames, h, w
            chunks = self._frames_to_chunks(frames)
            buffer = self.buffers.get(region)
            if buffer is None:
                scores.append(1.0)
                to_add[region] = chunks
                continue
            new = self.unseen(chunks, buffer)
            scores.append(0.0 if new is None else new.shape[0] / chunks.shape[0])
            if new is not None:
                to_add[region] = new
        if add:
            for region, chunks in to_add.items():
                old = self.buffers.get(region)
                self.buffers[region] = chunks if old is None else torch.cat([old, chunks], dim=0)
                if self.buffers[region].shape[0] > self.max_size:
                    self._rationalize(region)
        return max(scores) if scores else 0.0

    def copy(self) -> "OCRRegionBuffer":
        out = OCRRegionBuffer(n_chunks=self.n_chunks, max_size=self.max_size)
        out.buffers = {k: v.clone() for k, v in self.buffers.items()}
        return out

    def save(self, *, directory: str, merge: bool = False) -> None:
        os.makedirs(directory, exist_ok=True)
        for key, buffer in self.buffers.items():
            path = os.path.join(directory, f"ocr_buffer_{key}.pt")
            merged = buffer
            if merge and os.path.exists(path):
                existing = self.unseen(torch.load(path), buffer)
                if existing is not None:
                    merged = torch.cat([existing, buffer], dim=0)
            torch.save(merged.cpu(), path)

    @classmethod
    def load(cls, *, directory: str) -> Optional["OCRRegionBuffer"]:
        files = [f for f in os.listdir(directory) if f.startswith("ocr_buffer_") and f.endswith(".pt")]
        if not files:
            return None
        buf = cls()
        for f in files:
            buf.buffers[f[len("ocr_buffer_"):-len(".pt")]] = torch.load(os.path.join(directory, f))
            if buf.buffers[f[len("ocr_buffer_"):-len(".pt")]].shape[0] > buf.max_size:
                buf._rationalize(f[len("ocr_buffer_"):-len(".pt")])
        return buf


REGION_TYPES = {TEXT_LINES: TextLineBuffer, OCR: OCRRegionBuffer}
