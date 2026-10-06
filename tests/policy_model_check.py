"""
Checks that a policy model loads, generates, and backpropagates through VLMPolicy, and times it (GPU).
    python tests/policy_model_check.py --policy_model <model> [--policy_model <model> ...]
"""
import time
import click
import numpy as np
import torch


@click.command()
@click.option("--policy_model", "policy_models", multiple=True, required=True)
@click.option("--n_generate", default=5)
def main(policy_models, n_generate):
    from cusi.explore.policy import VLMPolicy
    try:
        import fla  # noqa: F401
        print("flash-linear-attention: installed", flush=True)
    except ImportError as e:
        print(f"flash-linear-attention: NOT importable ({e})", flush=True)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(144, 160, 3), dtype=np.uint8)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "You control a GameBoy. Reply with exactly one button from: UP, DOWN, LEFT, RIGHT, A, B, START."},
        {"type": "image", "image": 0}]}]
    results = {}
    for name in policy_models:
        print(f"\n=== {name}", flush=True)
        t = time.time()
        policy = VLMPolicy(model_name=name, max_new_tokens=32)
        print(f"  load {time.time() - t:.1f}s; LoRA targets {policy.lora_targets}", flush=True)
        out = policy.act(messages=messages, images=[frame])   # warm-up
        times, tokens = [], []
        for _ in range(n_generate):
            torch.cuda.synchronize()
            t = time.time()
            out = policy.act(messages=messages, images=[frame])
            torch.cuda.synchronize()
            times.append(time.time() - t)
            tokens.append(len(out["action_ids"]))
        print(f"  sample reply: {out['text']!r}", flush=True)
        print(f"  act(): {np.mean(times):.2f}s per call, {np.mean(tokens):.1f} tokens, "
              f"{np.sum(tokens) / np.sum(times):.1f} tokens/s (generate + scoring + ref)", flush=True)
        assert "<think>" not in out["text"], "thinking block in the reply: the non-thinking template was not used"
        # act's per-token log-probs must match evaluate's, or the PPO ratio is off before any update.
        long_msgs = [{"role": "user", "content": [{"type": "text", "text": "Describe this image in detail."},
                                                  {"type": "image", "image": 0}]}]
        policy.max_new_tokens = 48
        out_long = policy.act(messages=long_msgs, images=[frame])
        ev_train = policy.evaluate(messages=long_msgs, images=[frame], action_ids=out_long["action_ids"])
        with torch.no_grad():
            policy.model.eval()
            inputs = policy._encode(messages=long_msgs, images=[frame], response_ids=out_long["action_ids"])
            ev_eval = policy._forward(inputs=inputs, n_response=len(out_long["action_ids"]))
        d_train = (ev_train["token_logprob"].detach().float().cpu() - out_long["token_logprobs"]).abs()
        d_eval = (ev_eval["token_logprob"].float().cpu() - out_long["token_logprobs"]).abs()
        print(f"  {len(out_long['action_ids'])}-token reply: |act - evaluate(train mode)| per token mean "
              f"{float(d_train.mean()):.4f} max {float(d_train.max()):.4f}; |act - eval mode| mean "
              f"{float(d_eval.mean()):.4f} max {float(d_eval.max()):.4f}", flush=True)
        policy.max_new_tokens = 32
        ev = policy.evaluate(messages=messages, images=[frame], action_ids=out["action_ids"])
        loss = -ev["logprob"] + ev["value"] ** 2
        loss.backward()
        n_grad = sum(1 for p in policy.trainable_parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"  backward ok: {n_grad} trainable tensors got gradient; logprob {float(ev['logprob']):.3f} "
              f"(act said {out['logprob']:.3f})", flush=True)
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        results[name] = np.sum(tokens) / np.sum(times)
        del policy
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("\nSUMMARY tokens/s: " + ", ".join(f"{k}: {v:.1f}" for k, v in results.items()), flush=True)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
