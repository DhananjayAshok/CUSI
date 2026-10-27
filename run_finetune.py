"""
LoRA fine-tuning of the agent's model on a practice run's success rows (cusi.finetune), FSDP2 over every GPU, into
<model_dir>/finetune/<env>/<source>/<run_name>/ (adapter/ is what vLLM serves). Rerunning continues from the checkpoint.

    torchrun --standalone --nproc_per_node <gpus> run_finetune.py --env gameboy --model_name google/gemma-4-31b-it \
        --source tree_random_v1 --run_name lora_r16
"""
import os
import click
from cusi.agents.specs import ENV_NAMES
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import PracticePaths, finetune_run
from cusi.utils.run_dir import open_run_dir

loaded_parameters = load_parameters()
ROLES = ("supervisor", "executor")
OPERATIONAL_KEYS = {"log_every", "eval_every", "save_every", "num_workers", "max_batch_tokens"}


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--model_name", required=True, help="The model to fine-tune (an HF id); its practice run is the data.")
@click.option("--practice_model", default=None,
              help="The model whose practice run gives the data, if not --model_name.")
@click.option("--source", required=True, help="The practice run's task source (an annotated search run).")
@click.option("--run_name", required=True)
@click.option("--roles", default=",".join(ROLES), show_default=True, help="Which calls' rows to train on.")
@click.option("--lora_r", default=16, show_default=True)
@click.option("--lora_alpha", default=32, show_default=True)
@click.option("--lora_dropout", default=0.05, show_default=True)
@click.option("--lr", default=1e-4, show_default=True)
@click.option("--weight_decay", default=0.0, show_default=True)
@click.option("--epochs", default=1, show_default=True)
@click.option("--rows_per_step", default=32, show_default=True, help="Rows per optimizer step (all GPUs together).")
@click.option("--max_batch_tokens", default=8192, show_default=True,
              help="Padded tokens (rows x longest) per forward pass on one GPU; size it to the GPU's memory.")
@click.option("--warmup_frac", default=0.03, show_default=True)
@click.option("--max_length", default=4096, show_default=True, help="Longer rows are dropped.")
@click.option("--max_rows", default=None, type=int, help="Train on only the first rows of the shuffled order.")
@click.option("--val_rows", default=200, show_default=True, help="Validation rows sampled for the loss.")
@click.option("--log_every", default=5, show_default=True, help="Optimizer steps between log lines.")
@click.option("--eval_every", default=100, show_default=True, help="Optimizer steps between validations.")
@click.option("--save_every", default=50, show_default=True, help="Optimizer steps between checkpoints.")
@click.option("--seed", default=loaded_parameters["random_seed"], show_default=True)
@click.option("--num_workers", default=4, show_default=True, help="DataLoader workers encoding micro-batches per GPU.")
@click.option("--compile", "compile_layers", is_flag=True, default=False,
              help="torch.compile every decoder layer (experimental).")
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--ignore_config_violation", is_flag=True, default=False)
def main(env_name, model_name, practice_model, source, run_name, roles, overwrite, ignore_config_violation,
         compile_layers, **kwargs):
    from cusi.finetune.train import LoRATrainer
    parameters = loaded_parameters
    roles = [r.strip() for r in roles.split(",")]
    unknown = [r for r in roles if r not in ROLES]
    if unknown:
        raise click.UsageError(f"unknown roles {unknown}; roles are {ROLES}")
    practice = PracticePaths(parameters=parameters, env_name=env_name, model_name=practice_model or model_name,
                             source=source)
    config = {"env": env_name, "model_name": model_name, "dataset_dir": practice.dataset_dir, "roles": roles,
              "compile": compile_layers, **kwargs}
    out_dir = finetune_run(parameters=parameters, env=env_name, source=source, run=run_name)
    if int(os.environ.get("RANK", 0)) == 0:     # the other ranks wait for it at the first collective
        open_run_dir(directory=out_dir, config=config, overwrite=overwrite,
                     ignore_config_violation=ignore_config_violation, operational_keys=OPERATIONAL_KEYS)
        log_info(f"run_finetune: {model_name} on {practice.dataset_dir} -> {out_dir}", parameters=parameters)
    LoRATrainer(model_name=model_name, dataset_dir=practice.dataset_dir, out_dir=out_dir, config=config,
                parameters=parameters).train()


if __name__ == "__main__":
    main()
