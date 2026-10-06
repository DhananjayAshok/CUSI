"""
The exploration policy: a LoRA'd VLM with a value head, whose action is its generated text (PPO).
"""
import os
import shutil
from typing import Any, Optional
import torch
import torch.nn as nn
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info, log_warn
from cusi.agents.records import to_pil

def _disable_fla_without_compiler(*, parameters: dict) -> None:
    """Without a C compiler, hide flash-linear-attention from transformers; must run before the modeling module is imported."""
    if os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc"):
        return
    import transformers.utils.import_utils as import_utils
    import_utils.is_flash_linear_attention_available = lambda: False
    log_warn("No C compiler: flash-linear-attention disabled, using the torch implementation of linear attention",
             parameters=parameters)


# Never LoRA'd: the vision tower and the output head.
_NO_LORA = ("visual", "vision", "lm_head", "embed", "merger")


def lora_targets(*, model: nn.Module) -> list:
    """Leaf names of every nn.Linear in the language model, so LoRA covers whatever layer types it has."""
    names = set()
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(part in full_name for part in _NO_LORA):
            names.add(full_name.rsplit(".", 1)[-1])
    return sorted(names)


class VLMPolicy(nn.Module):
    def __init__(self, *, model_name: str, lora_r: int = 128, lora_alpha: int = 256,
                 lora_dropout: float = 0.0, temperature: float = 0.7, max_new_tokens: int = 256,
                 max_pixels: int = 640 * 28 * 28, device: str = None, gradient_checkpointing: bool = True,
                 parameters: dict[str, Any] = None) -> None:
        super().__init__()
        self._config = load_parameters(parameters)
        _disable_fla_without_compiler(parameters=self._config)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, get_peft_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        base = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype)
        if gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        self.lora_targets = lora_targets(model=base)
        config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                            target_modules=self.lora_targets, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, config).to(self.device)
        text_config = getattr(base.config, "text_config", base.config)
        hidden = text_config.hidden_size
        self.value_head = nn.Linear(hidden, 1).to(self.device)
        nn.init.normal_(self.value_head.weight, std=1e-3)
        nn.init.zeros_(self.value_head.bias)
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_info(f"VLMPolicy: {model_name} + LoRA r={lora_r}/alpha={lora_alpha} on {self.lora_targets} ({self.device});"
                 f" {n_train / 1e6:.1f}M trainable params",
                 parameters=self._config)

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _append_tokens(*, inputs: dict, ids: torch.Tensor) -> None:
        """Append text token ids (1, n) to every per-token tensor of the model inputs."""
        n = inputs["input_ids"].shape[1]
        for key, value in list(inputs.items()):
            if key in ("input_ids", "attention_mask") or not torch.is_tensor(value) or value.dim() != 2 \
                    or value.shape[1] != n:
                continue
            # other per-token tensors (e.g. mm_token_type_ids): appended tokens are text (0)
            inputs[key] = torch.cat([value, torch.zeros_like(ids, dtype=value.dtype)], dim=1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(ids)], dim=1)

    def _encode(self, *, messages: list, images: list, response_ids: Optional[torch.Tensor] = None,
                response_prefix: Optional[str] = None) -> dict:
        """Chat -> model inputs; response_prefix (prefilled reply start) counts as prompt, response_ids do not."""
        conv = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                conv.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
            else:
                conv.append({"role": msg["role"], "content": [
                    {"type": "image", "image": to_pil(images[p["image"]])} if p["type"] == "image"
                    else {"type": "text", "text": p["text"]} for p in content]})
        # enable_thinking=False: the non-thinking template (Qwen3.5); other templates ignore it.
        inputs = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt", enable_thinking=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        if response_prefix:
            prefix_ids = self.processor.tokenizer(response_prefix, add_special_tokens=False,
                                                  return_tensors="pt")["input_ids"].to(self.device)
            self._append_tokens(inputs=inputs, ids=prefix_ids)
        prompt_len = inputs["input_ids"].shape[1]
        if response_ids is not None:
            self._append_tokens(inputs=inputs, ids=response_ids.to(self.device).view(1, -1))
        inputs["prompt_len"] = prompt_len
        return inputs

    def _forward(self, *, inputs: dict, n_response: int, with_ref: bool = False) -> dict:
        prompt_len = inputs.pop("prompt_len")
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        logits = out.logits[0, prompt_len - 1: prompt_len - 1 + n_response].float() / self.temperature
        targets = inputs["input_ids"][0, prompt_len: prompt_len + n_response]
        logp_all = torch.log_softmax(logits, dim=-1)
        token_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        value = self.value_head(out.hidden_states[-1][0, prompt_len - 1].float()).squeeze(-1)
        result = {"logprob": token_logp.sum(), "token_logprob": token_logp, "entropy": entropy, "value": value,
                  "n_tokens": n_response}
        if with_ref:
            with torch.no_grad(), self.model.disable_adapter():
                ref = self.model(**inputs, use_cache=False)
                ref_logits = ref.logits[0, prompt_len - 1: prompt_len - 1 + n_response].float() / self.temperature
                result["ref_token_logprob"] = torch.log_softmax(ref_logits, dim=-1).gather(
                    -1, targets.unsqueeze(-1)).squeeze(-1)
                result["ref_logprob"] = result["ref_token_logprob"].sum()
        inputs["prompt_len"] = prompt_len
        return result

    # ------------------------------------------------------------------ API

    @torch.no_grad()
    def act(self, *, messages: list, images: list, response_prefix: Optional[str] = None) -> dict:
        """Sample an action and return its text, token ids, log-probs (policy and reference) and value."""
        self.model.eval()
        inputs = self._encode(messages=messages, images=images, response_prefix=response_prefix)
        prompt_len = inputs.pop("prompt_len")
        generated = self.model.generate(**inputs, do_sample=True, temperature=self.temperature, top_p=1.0, top_k=0,
                                        max_new_tokens=self.max_new_tokens,
                                        pad_token_id=self.processor.tokenizer.pad_token_id)
        action_ids = generated[0, prompt_len:]
        eos = self.processor.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.processor.tokenizer.pad_token_id}
        # keep the first end-of-turn token (it is part of the sampled action), drop padding after it
        cut = len(action_ids)
        for i, t in enumerate(action_ids.tolist()):
            if t in eos_ids:
                cut = i + 1
                break
        action_ids = action_ids[:cut].cpu()
        text = (response_prefix or "") + self.processor.tokenizer.decode(action_ids, skip_special_tokens=True)
        inputs = self._encode(messages=messages, images=images, response_ids=action_ids,
                              response_prefix=response_prefix)
        res = self._forward(inputs=inputs, n_response=len(action_ids), with_ref=True)
        return {"text": text, "action_ids": action_ids, "logprob": float(res["logprob"]),
                "value": float(res["value"]), "ref_logprob": float(res["ref_logprob"]),
                "token_logprobs": res["token_logprob"].detach().float().cpu(),
                "ref_token_logprobs": res["ref_token_logprob"].detach().float().cpu()}

    def evaluate(self, *, messages: list, images: list, action_ids: torch.Tensor, with_ref: bool = False,
                 response_prefix: Optional[str] = None) -> dict:
        """Re-encode one stored step and score its action (with grad)."""
        self.model.train()
        inputs = self._encode(messages=messages, images=images, response_ids=action_ids,
                              response_prefix=response_prefix)
        return self._forward(inputs=inputs, n_response=len(action_ids), with_ref=with_ref)

    @torch.no_grad()
    def value(self, *, messages: list, images: list, response_prefix: Optional[str] = None) -> float:
        self.model.eval()
        inputs = self._encode(messages=messages, images=images, response_prefix=response_prefix)
        prompt_len = inputs.pop("prompt_len")
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        return float(self.value_head(out.hidden_states[-1][0, prompt_len - 1].float()))

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, *, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        torch.save(self.value_head.state_dict(), os.path.join(directory, "value_head.pt"))

    def load(self, *, directory: str) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(self.model, load_file(os.path.join(directory, "adapter_model.safetensors")))
        self.value_head.load_state_dict(torch.load(os.path.join(directory, "value_head.pt"), map_location=self.device))
