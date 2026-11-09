"""
Diagnostics of practice adapters.

    python run_diagnostics.py action_repro --base_url http://127.0.0.1:8123/v1/ --base_model google/gemma-4-31b-it \
        --lora_model <served adapter name> --adapter <adapter dir> --out <json>
"""
import json
import os
import click
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info


@click.group()
@click.pass_context
def main(ctx):
    ctx.obj = load_parameters()


@main.command(name="action_repro")
@click.option("--base_url", required=True)
@click.option("--base_model", required=True, help="The served base model name.")
@click.option("--lora_model", required=True, help="The served LoRA module name.")
@click.option("--adapter", required=True, help="Adapter dir (its fine-tune config.json names the dataset).")
@click.option("--out", required=True)
@click.option("--n", default=300, help="Rows per split.")
@click.option("--workers", default=16)
@click.option("--max_tokens", default=2048)
@click.pass_obj
def action_repro(parameters, base_url, base_model, lora_model, adapter, out, n, workers, max_tokens):
    """Base vs adapter, greedy, on the adapter's own training rows and held-out practice rows."""
    from cusi.finetune.repro import run
    config = json.load(open(os.path.join(os.path.dirname(adapter.rstrip("/")), "config.json")))
    summary = run(base_url=base_url, base_model=base_model, lora_model=lora_model, dataset_dir=config["dataset_dir"],
                  n=n, out=out, workers=workers, max_tokens=max_tokens)
    log_info(f"action_repro {adapter}: {json.dumps(summary)}", parameters=parameters)


if __name__ == "__main__":
    main()
