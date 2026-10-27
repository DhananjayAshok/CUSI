"""
LoRA training of a multimodal chat model on practice success rows, one process per GPU under torchrun, FSDP2-sharded
(cusi.finetune.distributed). The adapters cover the language model's attention and MLP projections only (vLLM serves
LoRA on the language model, not the vision tower).

An optimizer step takes rows_per_step rows, packed by length into micro-batches of at most max_batch_tokens padded
tokens and dealt to the ranks (cusi.finetune.data.step_plan), encoded ahead in DataLoader workers. The loss is the mean over
the step's response tokens across all ranks (known in advance from the length index, so no extra collective).

The run directory holds config.json, lengths_validation.json (the train rows' length index is shared, in
<dataset>/lengths/), metrics.jsonl (one row per log interval and
per validation), checkpoint/ (sharded LoRA weights and optimizer state, rewritten every save_every steps, plus
trainer_state.json: a rerun continues from it), adapter/ (the final adapter, what vLLM serves) and compute.jsonl (one
row per job).
"""
import json
import math
import os
import time
from typing import Any
import torch
from cusi.finetune import distributed as fsdp
from cusi.finetune.data import (MicroBatches, RowEncoder, dataset_paths, keep_rows, label_tokens, length_index,
                                read_rows, row_order, step_plan)
from cusi.utils.compute import gpu_work, run_info, tally
from cusi.utils.log_handling import log_info

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def to_device(*, batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if v.is_floating_point():
                v = v.to(torch.bfloat16)
        out[k] = v
    return out


class LoRATrainer:
    def __init__(self, *, model_name: str, dataset_dir: str, out_dir: str, config: dict,
                 parameters: dict[str, Any]) -> None:
        """config: roles, lora_r, lora_alpha, lora_dropout, lr, weight_decay, epochs, rows_per_step, max_batch_tokens,
        warmup_frac, max_length, max_rows, val_rows, log_every, eval_every, save_every, seed, num_workers, compile
        (run_finetune.py)."""
        self._parameters = parameters
        self.model_name, self.dataset_dir, self.out_dir, self.config = model_name, dataset_dir, out_dir, config
        self.ckpt_dir = os.path.join(out_dir, "checkpoint")
        self.adapter_dir = os.path.join(out_dir, "adapter")
        self.rank, self.world, self.device = 0, 1, None

    def _info(self, message: str) -> None:
        if self.rank == 0:
            log_info(message, parameters=self._parameters)

    def _log(self, *, row: dict) -> None:
        if self.rank == 0:
            with open(os.path.join(self.out_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------ data

    def _data(self) -> None:
        from transformers import AutoProcessor
        c = self.config
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        encoder = RowEncoder(processor=self.processor)
        paths = dataset_paths(dataset_dir=self.dataset_dir)
        roles = tuple(c["roles"])
        self.train_rows = read_rows(path=paths["train"], roles=roles)
        val = read_rows(path=paths["validation"], roles=roles)
        g = torch.Generator().manual_seed(c["seed"])
        self.val_rows = [val[i] for i in torch.randperm(len(val), generator=g)[:c["val_rows"]].tolist()]
        t0 = time.time()
        # The train rows' index is shared by every run on this dataset, model and roles (next to the dataset); the
        # sampled validation rows' is the run's own.
        shared = os.path.join(self.dataset_dir, "lengths")
        if self.rank == 0:
            os.makedirs(shared, exist_ok=True)
        key = f"{self.model_name.split('/')[-1]}_{'-'.join(roles)}"
        paths = {"train": os.path.join(shared, f"{key}_train.json"),
                 "validation": os.path.join(self.out_dir, "lengths_validation.json")}
        index = {}
        for split, rows in (("train", self.train_rows), ("validation", self.val_rows)):
            index[split] = length_index(rows=rows, encoder=encoder, rank=self.rank, world=self.world,
                                        path=paths[split], gather=fsdp.gather_objects)
        self.train_lengths, self.val_lengths = index["train"], index["validation"]
        keep = keep_rows(lengths=self.train_lengths, max_length=c["max_length"])
        order = [keep[i] for i in row_order(n=len(keep), epochs=c["epochs"], seed=c["seed"])]
        if c["max_rows"] is not None:
            order = order[:c["max_rows"]]
        if c["max_batch_tokens"] < c["max_length"]:
            raise ValueError(f"max_batch_tokens {c['max_batch_tokens']} < max_length {c['max_length']}: the longest "
                             "rows would not fit in a micro-batch")
        self.rows_per_step = c["rows_per_step"]
        self.plan = step_plan(order=order, lengths=self.train_lengths, rows_per_step=self.rows_per_step,
                              max_batch_tokens=c["max_batch_tokens"], world=self.world)
        val_keep = keep_rows(lengths=self.val_lengths, max_length=c["max_length"])
        n_val = len(val_keep)
        self.val_plan = step_plan(order=val_keep, lengths=self.val_lengths, rows_per_step=n_val,
                                  max_batch_tokens=c["max_batch_tokens"], world=self.world) if n_val else []
        passes = sum(len(step[0]) for step in self.plan)
        mean_len = sum(self.train_lengths[i][1] for i in order) / max(1, len(order))
        self._info(f"finetune: {len(self.train_rows)} train rows ({', '.join(roles)}), "
                   f"{len(self.train_rows) - len(keep)} over {c['max_length']} tokens dropped; {len(self.plan)} "
                   f"optimizer steps of {self.rows_per_step} rows on {self.world} GPUs ({passes / max(1, len(self.plan)):.1f} "
                   f"micro-batches of <= {c['max_batch_tokens']} tokens per GPU per step), mean length {mean_len:.0f}; "
                   f"{n_val} validation rows (length index "
                   f"{time.time() - t0:.0f}s)")

    def _loader(self, *, rows: list, lengths: list, micro_batches: list):
        ds = MicroBatches(rows=rows, lengths=lengths, micro_batches=micro_batches, processor=self.processor)
        workers = self.config["num_workers"]
        return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, pin_memory=True,
                                           prefetch_factor=4 if workers else None)

    # ------------------------------------------------------------ model

    def _model(self) -> None:
        from peft import LoraConfig
        c = self.config
        lora = LoraConfig(r=c["lora_r"], lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        t0 = time.time()
        self.wrapper = fsdp.build(model_name=self.model_name, lora_config=lora, rank=self.rank, device=self.device,
                                  activation_checkpointing=True, compile_layers=c["compile"])
        params = list(fsdp.trainable(wrapper=self.wrapper).values())
        n = sum(p.numel() for p in params)
        self._info(f"finetune: model sharded over {self.world} GPUs in {time.time() - t0:.0f}s; {n / 1e6:.1f}M "
                   f"trainable parameters; {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB allocated per GPU")
        self.optimizer = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"], fused=True)
        total = len(self.plan)
        warmup = max(1, int(c["warmup_frac"] * total))

        def schedule(step: int) -> float:     # linear warmup, then cosine to 0
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.step = 0
        state_path = os.path.join(self.ckpt_dir, "trainer_state.json")
        if os.path.exists(state_path) and fsdp.load_checkpoint(wrapper=self.wrapper, optimizer=self.optimizer,
                                                               path=self.ckpt_dir):
            with open(state_path) as f:
                state = json.load(f)
            self.step = state["step"]
            self.scheduler.load_state_dict(state["scheduler"])
            self._info(f"finetune: resumed at step {self.step}/{total}")

    def _save(self) -> None:
        fsdp.save_checkpoint(wrapper=self.wrapper, optimizer=self.optimizer, path=self.ckpt_dir, rank=self.rank)
        if self.rank == 0:
            with open(os.path.join(self.ckpt_dir, "trainer_state.json"), "w") as f:
                json.dump({"step": self.step, "scheduler": self.scheduler.state_dict()}, f)
        torch.distributed.barrier()

    # ------------------------------------------------------------ loop

    def validate(self) -> float:
        if not self.val_plan:
            return float("nan")
        self.wrapper.eval()
        total = tokens = 0.0
        with torch.no_grad():
            for batch, n, weight in self._loader(rows=self.val_rows, lengths=self.val_lengths,
                                                 micro_batches=self.val_plan[0][self.rank]):
                total += float(self.wrapper(to_device(batch=batch, device=self.device))) * weight
                tokens += n * weight
        self.wrapper.train()
        total, tokens = fsdp.all_sum([total, tokens], self.device)
        return total / max(1.0, tokens)

    def train(self) -> dict:
        self.rank, self.world, self.device = fsdp.init()
        c = self.config
        with tally() as compute:
            self._data()
            self._model()
            self.wrapper.train()
            if self.step == 0:
                self._log(row={"step": 0, "val_loss": self.validate()})
            mine = [mb for step in self.plan[self.step:] for mb in step[self.rank]]
            batches = iter(self._loader(rows=self.train_rows, lengths=self.train_lengths, micro_batches=mine))
            fresh = {"loss": 0.0, "tokens": 0.0, "rows": 0, "seq_tokens": 0.0, "steps": 0, "wait": 0.0}
            t0, window = time.time(), dict(fresh)
            torch.cuda.reset_peak_memory_stats()
            while self.step < len(self.plan):
                step_rows = [i for rank_mbs in self.plan[self.step] for mb, w in rank_mbs if w for i in mb]
                step_tokens = label_tokens(lengths=self.train_lengths, idx=step_rows)
                loss_sum = 0.0
                with gpu_work("finetune_step"):
                    for _ in range(len(self.plan[self.step][self.rank])):
                        tw = time.time()
                        batch, _n, weight = next(batches)
                        window["wait"] += time.time() - tw
                        loss = self.wrapper(to_device(batch=batch, device=self.device))
                        # FSDP averages gradients over ranks: x world makes the result the token mean of the step
                        # (weight 0: a pass that only keeps the ranks in step).
                        (loss * (weight * self.world / step_tokens)).backward()
                        loss_sum += float(loss.detach()) * weight
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                window["loss"] += loss_sum
                window["tokens"] += step_tokens / self.world
                window["rows"] += len(step_rows)
                window["seq_tokens"] += sum(self.train_lengths[i][1] for i in step_rows)
                window["steps"] += 1
                if self.step % c["log_every"] == 0 or self.step == len(self.plan):
                    loss_total, tok_total = fsdp.all_sum([window["loss"], window["tokens"]], self.device)
                    dt = time.time() - t0
                    gn = grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm
                    row = {"step": self.step, "loss": loss_total / max(1.0, tok_total),
                           "lr": self.scheduler.get_last_lr()[0], "grad_norm": float(gn),
                           "seconds_per_step": dt / window["steps"], "rows_per_second": window["rows"] / dt,
                           "tokens_per_second": window["seq_tokens"] / dt, "data_wait_seconds": window["wait"],
                           "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
                    self._info(f"finetune step {self.step}/{len(self.plan)}: loss {row['loss']:.4f}, grad norm "
                               f"{row['grad_norm']:.3f}, {row['seconds_per_step']:.1f}s/step, "
                               f"{row['rows_per_second']:.2f} rows/s, {row['tokens_per_second']:.0f} tokens/s, "
                               f"data wait {row['data_wait_seconds']:.1f}s, peak {row['peak_gb']:.1f} GB")
                    self._log(row=row)
                    t0, window = time.time(), dict(fresh)
                if self.step % c["eval_every"] == 0:
                    val = self.validate()
                    self._info(f"finetune step {self.step}: validation loss {val:.4f}")
                    self._log(row={"step": self.step, "val_loss": val})
                if self.step % c["save_every"] == 0:
                    self._save()
            val = self.validate()
            self._log(row={"step": self.step, "val_loss": val, "final": True})
            self._save()
            fsdp.save_adapter(wrapper=self.wrapper, path=self.adapter_dir, rank=self.rank)
        summary = {"steps": self.step, "final_val_loss": val, "world": self.world, "compute": compute.snapshot(),
                   **run_info()}
        if self.rank == 0:
            with open(os.path.join(self.out_dir, "compute.jsonl"), "a") as f:
                f.write(json.dumps(summary, default=str) + "\n")
        self._info(f"finetune: done, validation loss {val:.4f}; adapter -> {self.adapter_dir}")
        torch.distributed.destroy_process_group()
        return summary
