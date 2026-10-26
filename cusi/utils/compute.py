"""
Compute accounting: wall time of the parts of a unit of work (a search step, an expansion, a practice task), the local
GPU work in it (GPU-seconds by device type and FLOPs), its LLM calls (latency, tokens, FLOPs) and what NVML sees on the
job's GPUs over it (energy and busy time), which includes a vLLM server sharing the node.

    with tally() as t:                  # collects what this thread records until it closes
        with part("env"):               # wall seconds of a named part
            ...
        with gpu_work("embed", key="one"):   # local torch work: GPU-seconds, and FLOPs (counted on the first call
            ...                              # per (name, key), so key must fix the work's size)
        record = t.snapshot()           # JSON-ready totals so far

Tallies nest per thread: a record goes to every open tally of the thread it is made on, and to the process total
(TOTAL). LLM calls are recorded by AgentVLM; their FLOPs are worked out from the served model's own config and weights
(llm_flops) as if no prompt prefix were cached, so the vLLM server's real work can be lower (NVML shows it).
"""
import json
import os
import socket
import struct
import sys
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

_local = threading.local()
_flops_cache: dict = {}          # (gpu_work name, key) -> FLOPs of one call
_lock = threading.Lock()


# --------------------------------------------------------------------------- tallies

def _empty() -> dict:
    return {"wall_s": {}, "gpu": {}, "llm": {"calls": 0, "records": 0, "latency_s": 0.0, "prompt_tokens": 0,
                                             "completion_tokens": 0, "images": 0, "flops": 0.0,
                                             "flops_unknown_calls": 0}}


class Tally:
    """Totals of what one thread records while the tally is open, plus the NVML deltas over it."""

    def __init__(self) -> None:
        self.data = _empty()
        self._t0 = time.perf_counter()
        self._nvml0 = _nvml.read()

    def snapshot(self) -> dict:
        """JSON-ready totals so far: wall_s parts and "total", gpu sections, llm, and nvml per device."""
        with _lock:
            out = json.loads(json.dumps(self.data))
        out["wall_s"]["total"] = time.perf_counter() - self._t0
        nvml = _nvml.delta(self._nvml0)
        if nvml:
            out["nvml"] = nvml
        return out


class _Total(Tally):
    """The process total (every thread); TOTAL is made once NVML is set up, below."""


def _open() -> list:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def _targets() -> list:
    return [TOTAL.data] + [t.data for t in _open()]


@contextmanager
def tally():
    t = Tally()
    stack = _open()
    stack.append(t)
    try:
        yield t
    finally:
        stack.remove(t)


@contextmanager
def part(name: str):
    """Wall seconds of a named part, added to every open tally."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _lock:
            for d in _targets():
                d["wall_s"][name] = d["wall_s"].get(name, 0.0) + dt


# --------------------------------------------------------------------------- local GPU work

def _torch():
    """torch if this process imported it (no import otherwise: a CPU-only run never loads it)."""
    return sys.modules.get("torch")


def _cuda_allocs(torch) -> int:
    if torch is None or not torch.cuda.is_initialized():
        return 0
    return int(torch.cuda.memory_stats().get("allocation.all.allocated", 0))


@contextmanager
def gpu_work(name: str, *, key=None):
    """Time local torch work (synchronised) and count it as GPU work when it allocated on the GPU; work that stays
    on the CPU records nothing. FLOPs: counted with FlopCounterMode on the first call per (name, key) and reused
    after, so key must determine the work's size (an input shape); key None leaves them uncounted."""
    torch = _torch()
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    allocs0 = _cuda_allocs(torch)
    count = key is not None and torch is not None and (name, key) not in _flops_cache
    t0 = time.perf_counter()
    try:
        if count:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as counter:
                yield
            _flops_cache[(name, key)] = float(counter.get_total_flops())
        else:
            yield
    finally:
        torch = _torch()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if _cuda_allocs(torch) != allocs0:
                device = torch.cuda.get_device_name(torch.cuda.current_device())
                flops = _flops_cache.get((name, key)) if key is not None else None
                with _lock:
                    for d in _targets():
                        g = d["gpu"].setdefault(name, {"device": device, "calls": 0, "s": 0.0, "flops": 0.0,
                                                       "flops_unknown_calls": 0})
                        g["calls"] += 1
                        g["s"] += dt
                        if flops is None:
                            g["flops_unknown_calls"] += 1
                        else:
                            g["flops"] += flops


def shape_key(obj) -> tuple:
    """A hashable key of the tensor shapes in obj (nested dicts / lists / tuples), for gpu_work's key."""
    torch = _torch()
    if torch is not None and isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    if isinstance(obj, dict):
        return tuple((k, shape_key(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(shape_key(v) for v in obj)
    return ()


# --------------------------------------------------------------------------- LLM calls

def _hf_file(model_name: str, filename: str) -> Optional[str]:
    if os.path.isdir(model_name):
        path = os.path.join(model_name, filename)
        return path if os.path.exists(path) else None
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(model_name, filename)
    return path if isinstance(path, str) else None


def _safetensors_params(model_name: str) -> Optional[dict]:
    """Parameter counts by part ("vision", "embed", "head", "lm") from the cached weights' headers."""
    index = _hf_file(model_name, "model.safetensors.index.json")
    shards = sorted(set(json.load(open(index))["weight_map"].values())) if index else ["model.safetensors"]
    counts = {"vision": 0, "embed": 0, "head": 0, "lm": 0}
    for shard in shards:
        path = _hf_file(model_name, shard)
        if path is None:
            return None
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for tensor, meta in header.items():
            if tensor == "__metadata__":
                continue
            n = 1
            for s in meta["shape"]:
                n *= s
            low = tensor.lower()
            kind = ("vision" if ("vision" in low or "visual" in low) else "embed" if "embed_tokens" in low
                    else "head" if "lm_head" in low else "lm")
            counts[kind] += n
    return counts


@lru_cache(maxsize=None)
def model_shape(model_name: str) -> Optional[dict]:
    """What llm_flops needs, from the served model's cached config.json and weights; None if they are not cached."""
    config_path = _hf_file(model_name, "config.json")
    if config_path is None:
        return None
    config = json.load(open(config_path))
    text = config.get("text_config", config)
    params = _safetensors_params(model_name)
    if params is None:
        return None
    layers = int(text["num_hidden_layers"])
    heads = int(text["num_attention_heads"])
    head_dim = int(text.get("head_dim") or text["hidden_size"] // heads)
    types = text.get("layer_types") or ["full_attention"] * layers
    window = text.get("sliding_window")
    attn = [(heads * (int(text.get("global_head_dim") or head_dim) if t == "full_attention" else head_dim),
             None if t == "full_attention" or window is None else int(window)) for t in types]
    vision = config.get("vision_config")
    vision_shape = None
    if vision and params["vision"]:
        per_image = config.get("vision_soft_tokens_per_image") or vision.get("default_output_length")
        pool = vision.get("pooling_kernel_size") or 1
        if per_image:
            v_heads = int(vision["num_attention_heads"])
            vision_shape = {"params": params["vision"], "patches": int(per_image) * int(pool) ** 2,
                            "layers": int(vision["num_hidden_layers"]),
                            "width": v_heads * int(vision.get("head_dim") or vision["hidden_size"] // v_heads)}
    return {"lm_params": params["lm"], "hidden": int(text["hidden_size"]), "vocab": int(text["vocab_size"]),
            "attention": attn, "vision": vision_shape}


def _context_sum(n: int, window: Optional[int]) -> int:
    """sum over positions t = 1..n of the attended context min(t, window)."""
    if window is None or n <= window:
        return n * (n + 1) // 2
    return window * (window + 1) // 2 + (n - window) * window


def llm_flops(*, model_name: str, prompt_tokens: int, completion_tokens: int, images: int) -> Optional[float]:
    """Forward FLOPs of one request: 2 x the language model's weights per token, attention scores and values over
    each token's context (sliding-window layers capped), logits for each generated token, and the vision tower per
    image (its weights per patch, plus attention over the patches). No prefix cache is assumed."""
    shape = model_shape(model_name)
    if shape is None:
        return None
    n = prompt_tokens + completion_tokens
    flops = 2.0 * shape["lm_params"] * n
    flops += sum(4.0 * width * _context_sum(n, window) for width, window in shape["attention"])
    flops += 2.0 * shape["vocab"] * shape["hidden"] * completion_tokens
    v = shape["vision"]
    if images and v is not None:
        flops += images * (2.0 * v["params"] * v["patches"] + 4.0 * v["layers"] * v["width"] * v["patches"] ** 2)
    return flops


def record_llm(*, model_name: str, latency: float, prompt_tokens: list, completion_tokens: list,
               images: list) -> None:
    """One call (one or more records) to every open tally."""
    flops, unknown = 0.0, 0
    for p, c, i in zip(prompt_tokens, completion_tokens, images):
        f = llm_flops(model_name=model_name, prompt_tokens=p or 0, completion_tokens=c or 0, images=i) \
            if p is not None and c is not None else None
        if f is None:
            unknown = 1
        else:
            flops += f
    with _lock:
        for d in _targets():
            llm = d["llm"]
            llm["calls"] += 1
            llm["records"] += len(prompt_tokens)
            llm["latency_s"] += latency
            llm["prompt_tokens"] += sum(p or 0 for p in prompt_tokens)
            llm["completion_tokens"] += sum(c or 0 for c in completion_tokens)
            llm["images"] += sum(images)
            llm["flops"] += flops
            llm["flops_unknown_calls"] += unknown


# --------------------------------------------------------------------------- NVML (the job's GPUs)

class _NVML:
    """Energy counters (read directly) and busy time (GPU utilisation integrated by a sampling thread) of the job's
    GPUs, a vLLM server's included: those whose UUIDs CUSI_ACCOUNT_GPUS lists (set by scripts/utils.sh), else every GPU
    NVML can see. Silent without NVML (no driver, e.g. a login node)."""

    def __init__(self, period: float = 0.5) -> None:
        self.period = period
        self._handles = None
        self._busy: dict = {}
        self._started = False

    def _init(self) -> list:
        if self._handles is not None:
            return self._handles
        self._handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            wanted = {u for u in os.environ.get("CUSI_ACCOUNT_GPUS", "").split(",") if u}
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                if not wanted or pynvml.nvmlDeviceGetUUID(h) in wanted:
                    self._handles.append((i, pynvml.nvmlDeviceGetName(h), h))
        except Exception:
            self._handles = []
        if self._handles and not self._started:
            self._started = True
            self._busy = {i: 0.0 for i, _, _ in self._handles}
            threading.Thread(target=self._sample, daemon=True, name="nvml-busy").start()
        return self._handles

    def _sample(self) -> None:
        import pynvml
        last = time.perf_counter()
        while True:
            time.sleep(self.period)
            now = time.perf_counter()
            dt, last = now - last, now
            for i, _, h in self._handles:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu / 100.0
                except Exception:
                    continue
                self._busy[i] += util * dt

    def read(self) -> dict:
        """{index: (name, energy mJ or None, busy seconds)}."""
        if not self._init():
            return {}
        import pynvml
        out = {}
        for i, name, h in self._handles:
            try:
                energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            except Exception:
                energy = None
            out[i] = (name, energy, self._busy.get(i, 0.0))
        return out

    def delta(self, start: dict) -> dict:
        """{index: {"name", "energy_j", "busy_s"}} since a read()."""
        now = self.read()
        out = {}
        for i, (name, energy, busy) in now.items():
            if i not in start:
                continue
            e0 = start[i][1]
            out[str(i)] = {"name": name, "busy_s": round(busy - start[i][2], 3),
                           "energy_j": None if energy is None or e0 is None else round((energy - e0) / 1000.0, 3)}
        return out


_nvml = _NVML()
TOTAL = _Total()


def run_info() -> dict:
    """Where the work runs: host, Slurm job and the GPUs this process and NVML see."""
    torch = _torch()
    visible = []
    if torch is not None and torch.cuda.is_available():
        visible = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_gpus": visible,
            "nvml_gpus": {str(i): name for i, (name, _, _) in _nvml.read().items()},
            "nvml_scope": "CUSI_ACCOUNT_GPUS" if os.environ.get("CUSI_ACCOUNT_GPUS") else "all visible"}


def merge(*, into: dict, other: dict) -> dict:
    """Add one snapshot to another (both from Tally.snapshot); into is changed and returned."""
    for part_name, s in other.get("wall_s", {}).items():
        into.setdefault("wall_s", {})[part_name] = into.get("wall_s", {}).get(part_name, 0.0) + s
    for section, g in other.get("gpu", {}).items():
        acc = into.setdefault("gpu", {}).setdefault(section, {"device": g["device"], "calls": 0, "s": 0.0,
                                                              "flops": 0.0, "flops_unknown_calls": 0})
        for k in ("calls", "s", "flops", "flops_unknown_calls"):
            acc[k] += g[k]
    llm = into.setdefault("llm", _empty()["llm"])
    for k, v in other.get("llm", {}).items():
        llm[k] = llm.get(k, 0) + v
    for i, n in other.get("nvml", {}).items():
        acc = into.setdefault("nvml", {}).setdefault(i, {"name": n["name"], "busy_s": 0.0, "energy_j": 0.0})
        acc["busy_s"] += n["busy_s"]
        acc["energy_j"] = None if acc["energy_j"] is None or n["energy_j"] is None else acc["energy_j"] + n["energy_j"]
    return into


#: What a compute budget can be set in (budget_value).
BUDGET_METRICS = ("wall_s", "gpu_s", "llm_flops", "flops", "gpu_busy_s", "energy_j")


def budget_value(*, snapshot: dict, metric: str) -> float:
    """One snapshot's amount of a BUDGET_METRICS metric: wall seconds; local GPU seconds; LLM FLOPs; LLM plus local
    GPU FLOPs; NVML busy seconds or energy over all the job's GPUs (a vLLM server's included)."""
    gpu = snapshot.get("gpu", {}).values()
    nvml = snapshot.get("nvml", {}).values()
    if metric == "wall_s":
        return snapshot["wall_s"]["total"]
    if metric == "gpu_s":
        return sum(g["s"] for g in gpu)
    if metric == "llm_flops":
        return snapshot["llm"]["flops"]
    if metric == "flops":
        return snapshot["llm"]["flops"] + sum(g["flops"] for g in gpu)
    if metric == "gpu_busy_s":
        return sum(n["busy_s"] for n in nvml)
    if metric == "energy_j":
        return sum(n["energy_j"] or 0.0 for n in nvml)
    raise ValueError(f"unknown budget metric {metric!r}; one of {BUDGET_METRICS}")
