"""Test-set evaluation through cusi_envs in test mode (eval_plan.md, option 2).

    python run_eval.py gameboy --model_name <served name> --run_name dev [--n_tasks N]
    python run_eval.py android --model_name <served name> --run_name dev [--tasks A,B]
    python run_eval.py web --model_name <served name> --run_name dev [--tasks Site--0,...]

Outputs: storage_dir/eval/<env>/<run_name>/ (results.jsonl, summary.json, config.json, per-task
trajectories). Each command's options are in cusi_eval/<env>.py.
"""
import click
from cusi_utils.parameter_handling import load_parameters

loaded_parameters = load_parameters()


class LazyGroup(click.Group):
    """Commands import their benchmark only when run (GameBoyRL's and WebVoyager's top-level
    `utils` modules cannot share a process)."""
    COMMANDS = {"gameboy": "cusi_eval.gameboy", "android": "cusi_eval.android", "web": "cusi_eval.web"}

    def list_commands(self, ctx):
        return sorted(self.COMMANDS)

    def get_command(self, ctx, name):
        if name not in self.COMMANDS:
            return None
        import importlib
        return importlib.import_module(self.COMMANDS[name]).command


@click.group(cls=LazyGroup)
@click.pass_context
def main(ctx):
    ctx.obj = loaded_parameters


if __name__ == "__main__":
    main()
