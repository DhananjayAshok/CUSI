"""
Test-set evaluation through cusi.envs in test mode.

    python run_eval.py <gameboy|android|web> --model_name <served name> --run_name dev
"""
import click
from cusi.utils.parameter_handling import load_parameters

loaded_parameters = load_parameters()


class LazyGroup(click.Group):
    """Imports each benchmark only when run (GameBoyRL's and WebVoyager's `utils` modules clash)."""
    COMMANDS = {"gameboy": "cusi.eval.gameboy", "android": "cusi.eval.android", "web": "cusi.eval.web"}

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
