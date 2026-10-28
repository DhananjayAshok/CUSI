"""
The FSDP2 side of LoRA fine-tuning (one process per GPU, launched by torchrun), after torchtune's distributed LoRA
recipe and torchtitan:
- the model is built on the meta device on every rank; rank 0 alone reads the checkpoint (to CPU) and its weights and
  buffers are broadcast into the shards, so host memory holds one copy, not one per rank;
- every decoder / encoder layer (the model's _no_split_modules) is its own FSDP unit and the CompletionLoss wrapper
  the root; params are gathered in bf16, gradients reduced in fp32, the LoRA weights kept in fp32 (peft's default);
- the loss runs the transformer body, then the output projection and logit softcap only at the response positions
  (CompletionLoss), not over the 262k vocabulary at every position;
- checkpoints are torch.distributed.checkpoint shards of the LoRA weights and optimizer state (resumable on the same
  or another GPU count); the adapter vLLM serves is gathered to rank 0 at the end.
"""
import os
from typing import Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from cusi.finetune.data import IGNORE_INDEX


def init() -> tuple:
    """-> (rank, world size, device). Run under torchrun (or alone: rank 0 of 1)."""
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", local))
    return rank, world, torch.device("cuda", local)


def gather_objects(obj) -> list:
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def all_sum(values: list, device) -> list:
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


LOSS_CHUNK = 256      # labelled positions per logits chunk: 256 x 262k vocab x 4 B = 268 MB


class CompletionLoss(nn.Module):
    """The summed cross-entropy of a batch's labelled tokens. As Gemma4ForConditionalGeneration.forward (the body,
    then lm_head and the final logit softcap) but projecting only the positions whose next token has a label, in
    chunks of LOSS_CHUNK positions whose logits are recomputed in backward (torchtune's chunked cross-entropy), so the
    logits' memory does not grow with the response tokens of a micro-batch."""

    def __init__(self, *, model) -> None:
        super().__init__()
        self.model = model         # the PeftModel

    def _chunk_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.model.get_base_model()
        logits = base.lm_head(hidden)
        cap = base.config.get_text_config().final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return F.cross_entropy(logits.float(), targets, reduction="sum")

    def forward(self, batch: dict) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        base = self.model.get_base_model()
        labels = batch["labels"]
        hidden = base.model(**batch, use_cache=False, return_dict=True).last_hidden_state
        targets = labels[:, 1:]
        keep = targets != IGNORE_INDEX
        hidden, targets = hidden[:, :-1][keep], targets[keep]
        total = hidden.new_zeros((), dtype=torch.float32)
        for h, t in zip(hidden.split(LOSS_CHUNK), targets.split(LOSS_CHUNK)):
            total = total + checkpoint(self._chunk_loss, h, t, use_reentrant=False)
        return total


def _vision_position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor):
    """Gemma4VisionPatchEmbedder._position_embeddings as a table lookup. HF multiplies a one-hot int64 tensor
    (images x 2520 patches x 2 x 10240, ~394 MiB per image) by the table, so a micro-batch of 20+ images ran out of
    memory; selecting the rows gives bitwise the same result."""
    positions = pixel_position_ids.clamp(min=0)
    table = self.position_embedding_table
    embeddings = table[0][positions[..., 0]] + table[1][positions[..., 1]]
    return torch.where(padding_positions.unsqueeze(-1), 0.0, embeddings)


def patch_model_classes() -> None:
    """Memory fixes to HF model code, applied before the model is built (equal outputs)."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder
    except ImportError:
        return
    Gemma4VisionPatchEmbedder._position_embeddings = _vision_position_embeddings


def build(*, model_name: str, lora_config, rank: int, device, activation_checkpointing: bool,
          compile_layers: bool) -> CompletionLoss:
    """The sharded CompletionLoss(PeftModel) on device, weights loaded from model_name (read by rank 0 only)."""
    from peft import get_peft_model
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoConfig, AutoModelForImageTextToText
    patch_model_classes()
    config = AutoConfig.from_pretrained(model_name)
    full_sd, full_buffers = None, None
    if rank == 0:
        cpu = get_peft_model(AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16,
                                                                         device_map="cpu"), lora_config)
        full_sd = cpu.state_dict()
        full_buffers = dict(cpu.named_buffers())
    with torch.device("meta"):
        model = get_peft_model(AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16), lora_config)
    if activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    wrapper = CompletionLoss(model=model)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    unit_classes = set(model.get_base_model()._no_split_modules or [])
    units = [m for m in wrapper.modules() if type(m).__name__ in unit_classes]
    if not units:
        raise ValueError(f"no FSDP units: none of {sorted(unit_classes)} in the model")
    for m in units:
        if compile_layers:
            m.compile()
        fully_shard(m, mp_policy=mp)
    # The root (embeddings, tied lm_head: ~2.8 GB gathered) reshards after forward too: one more gather per pass,
    # but ~2.4 GB more room for the backward pass (on A5000s a ~3.7k-token row's backward asked for 1.7 GB more, the
    # size of a 32-head L x L attention matrix in fp32)
    fully_shard(wrapper, mp_policy=mp)
    wrapper.to_empty(device=device)
    base = model.get_base_model()
    if getattr(base, "_tied_weights_keys", None) and base.lm_head.weight is not base.get_input_embeddings().weight:
        raise RuntimeError("lm_head and the input embeddings were untied by the meta-device build")
    set_model_state_dict(wrapper.model, full_sd if rank == 0 else {},
                         options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    # Buffers (rotary inverse frequencies, ...) are mostly non-persistent: not in the state dict, and to_empty left
    # them uninitialised. Rank 0 has them all: one keyed broadcast (per-buffer broadcasts deadlocked when ranks
    # enqueued them out of step).
    box = [{n: b.clone() for n, b in full_buffers.items()} if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    with torch.no_grad():
        for name, buf in wrapper.model.named_buffers():
            if name not in box[0]:
                raise KeyError(f"rank {rank}: buffer {name} not in rank 0's model")
            buf.copy_(box[0][name].to(device))
    del full_sd, full_buffers, box
    return wrapper


def trainable(*, wrapper) -> dict:
    return {n: p for n, p in wrapper.model.named_parameters() if p.requires_grad}


def save_checkpoint(*, wrapper, optimizer, path: str, rank: int) -> None:
    """Sharded LoRA weights + optimizer state into path (replaced atomically)."""
    import shutil
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
    tmp = path + ".tmp"
    if rank == 0 and os.path.exists(tmp):
        shutil.rmtree(tmp)
    dist.barrier()
    dcp.save({"lora": trainable(wrapper=wrapper), "optim": get_optimizer_state_dict(wrapper.model, optimizer)},
             checkpoint_id=tmp)
    dist.barrier()
    if rank == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(tmp, path)
    dist.barrier()


def load_checkpoint(*, wrapper, optimizer, path: str) -> bool:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
    if not os.path.isdir(path):
        return False
    state = {"lora": trainable(wrapper=wrapper), "optim": get_optimizer_state_dict(wrapper.model, optimizer)}
    dcp.load(state, checkpoint_id=path)
    set_optimizer_state_dict(wrapper.model, optimizer, state["optim"])
    return True


def save_adapter(*, wrapper, path: str, rank: int) -> Optional[str]:
    """Gather the LoRA weights to rank 0 and write a peft adapter (adapter_config.json + adapter_model.safetensors)."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    full = {}
    for name, p in trainable(wrapper=wrapper).items():
        t = p.full_tensor()
        if rank == 0:
            full[name] = t.detach().to("cpu", torch.float32).contiguous()
    if rank != 0:
        return None
    model = wrapper.model
    os.makedirs(path, exist_ok=True)
    save_file(get_peft_model_state_dict(model, state_dict=full), os.path.join(path, "adapter_model.safetensors"),
              metadata={"format": "pt"})
    model.peft_config["default"].save_pretrained(path)
    return path
