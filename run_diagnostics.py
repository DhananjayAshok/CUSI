"""
Diagnostics of practice adapters.

    python run_diagnostics.py action_repro --base_url http://127.0.0.1:8123/v1/ --base_model google/gemma-4-31b-it \
        --lora_model <served adapter name> --adapter <adapter dir> --out <json>
    python run_diagnostics.py base_disagree --base_url <url> --model <served base> --dataset_dir <dir> --out <jsonl>
    python run_diagnostics.py base_decode_all --base_url <url> --model <served base> --env <env> \
        --dataset_dir <dir> --out <jsonl>
    python run_diagnostics.py disagree_dataset --src_dir <dir> --records <jsonl> --out_dir <new dataset dir>
    python run_diagnostics.py temperature_sweep --env <env> --runs <eval run dir,...> --base_url <url> \
        --model <served model> --out <jsonl>
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


@main.command(name="base_disagree")
@click.option("--base_url", required=True)
@click.option("--model", required=True, help="The served base model name.")
@click.option("--dataset_dir", required=True, help="A practice dataset dir (its success/train action rows).")
@click.option("--out", required=True, help="Records JSONL (appended; a rerun resumes).")
@click.option("--workers", default=16)
@click.pass_obj
def base_disagree(parameters, base_url, model, dataset_dir, out, workers):
    """The served model decodes every action row as the eval agent would; its normalised action vs the row's."""
    from cusi.finetune.disagree import decode_base
    records = decode_base(base_url=base_url, model=model, dataset_dir=dataset_dir, out=out, workers=workers)
    agree = sum(r["agree"] for r in records)
    log_info(f"base_disagree {dataset_dir}: {len(records)} action rows, base agrees on {agree}, disagrees on "
             f"{len(records) - agree} -> {out}", parameters=parameters)


@main.command(name="base_decode_all")
@click.option("--base_url", required=True)
@click.option("--model", required=True, help="The served base model name.")
@click.option("--env", "env_name", required=True, type=click.Choice(["android", "web", "gameboy"]))
@click.option("--dataset_dir", required=True)
@click.option("--out", required=True, help="Records JSONL (appended; a rerun resumes).")
@click.option("--n", default=None, type=int, help="A task-stratified sample of this many rows (default: all).")
@click.option("--seed", default=0, show_default=True)
@click.option("--workers", default=32)
@click.pass_obj
def base_decode_all(parameters, base_url, model, env_name, dataset_dir, out, n, seed, workers):
    """Every row (any role / tag) decoded by the served model as the eval agent would call it; raw replies kept."""
    from cusi.finetune.disagree import decode_all
    total = decode_all(base_url=base_url, model=model, env=env_name, dataset_dir=dataset_dir, out=out,
                       workers=workers, n=n, seed=seed)
    log_info(f"base_decode_all {env_name} {dataset_dir}: {total} rows -> {out}", parameters=parameters)


@main.command(name="disagree_dataset")
@click.option("--src_dir", required=True, help="The dataset the records were decoded on.")
@click.option("--records", required=True, help="base_disagree's records JSONL.")
@click.option("--out_dir", required=True, help="The new dataset dir (a practice source's dataset/).")
@click.option("--val_copy", default=200, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.pass_obj
def disagree_dataset(parameters, src_dir, records, out_dir, val_copy, seed):
    """A dataset of the base-disagreement action rows only (cusi.finetune.disagree)."""
    from cusi.finetune.disagree import build_disagreement_dataset
    stats = build_disagreement_dataset(src_dir=src_dir, out_dir=out_dir, records_path=records, val_copy=val_copy,
                                       seed=seed)
    log_info(f"disagree_dataset: {json.dumps(stats)} -> {out_dir}", parameters=parameters)


@main.command(name="temperature_sweep")
@click.option("--env", "env_name", required=True, type=click.Choice(["android", "web", "gameboy"]))
@click.option("--runs", required=True, help="Comma-separated eval run dirs whose leg records are replayed.")
@click.option("--base_url", required=True)
@click.option("--model", required=True, help="The served model name.")
@click.option("--out", required=True, help="Replies JSONL (appended; a rerun resumes); summary at <out>.summary.json.")
@click.option("--temperatures", default="0,0.5,1,1.3,1.6,2,2.5,3", show_default=True)
@click.option("--n_per_tag", default=40, show_default=True, help="Recorded calls sampled per call tag.")
@click.option("--samples", default=2, show_default=True, help="Replies per call and temperature.")
@click.option("--seed", default=0, show_default=True)
@click.option("--workers", default=32)
@click.pass_obj
def temperature_sweep(parameters, env_name, runs, base_url, model, out, temperatures, n_per_tag, samples, seed,
                      workers):
    """Recorded test-set calls replayed at each temperature through the env's parsers (cusi.eval.temperature_sweep)."""
    from cusi.eval.temperature_sweep import summarise, sweep
    records = sweep(env=env_name, run_dirs=runs.split(","), model_name=model, vllm_base_url=base_url,
                    temperatures=[float(t) for t in temperatures.split(",")], n_per_tag=n_per_tag, samples=samples,
                    out=out, seed=seed, workers=workers)
    rows = summarise(records=records)
    with open(out + ".summary.json", "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        log_info(json.dumps(r), parameters=parameters)


if __name__ == "__main__":
    main()
