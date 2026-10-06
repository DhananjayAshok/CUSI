"""
Prints a named string (e.g. an experiment name) built from --key value args, for use from bash.

Must print exactly once to stdout; errors go to stderr with a non-zero exit.
"""
import sys
from abc import ABC, abstractmethod

def depathify(string) -> str:
    """Make a path-like string safe to use as a filename or experiment name."""
    return (
        string.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

def log(message: str) -> None:
    print(message, file=sys.stderr)


class StringFunction(ABC):
    NAME = None # name used to call it from bash
    # Arg names are case insensitive and may not contain ' '
    REQUIRED_ARGS = []
    OPTIONAL_ARGS = {} # name -> default; unexpected args are ignored
    # Prefer shared optional args in scripts/utils.sh over OPTIONAL_ARGS here.

    def __init__(self):
        if self.NAME is None:
            raise ValueError("StringFunction must have a NAME attribute")
        if " " in self.NAME:
            raise ValueError(f"StringFunction NAME cannot contain spaces. Got: {self.NAME}")
        for i in range(len(self.REQUIRED_ARGS)):
            self.REQUIRED_ARGS[i] = self.REQUIRED_ARGS[i].lower()
            if " " in self.REQUIRED_ARGS[i]:
                raise ValueError(f"Argument names cannot contain spaces. Got: {self.REQUIRED_ARGS[i]}")
        for arg in self.OPTIONAL_ARGS:
            arg = arg.lower()
            if " " in arg:
                raise ValueError(f"Argument names cannot contain spaces. Got: {arg}")
            if arg in self.REQUIRED_ARGS:
                raise ValueError(f"Argument {arg} cannot be both required and optional.")


    def validate_args(self, **kwargs):
        for arg in self.REQUIRED_ARGS:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")
        for arg in kwargs:
            if arg not in self.REQUIRED_ARGS and arg not in self.OPTIONAL_ARGS:
                pass # unexpected arguments are ignored

    @abstractmethod
    def _get_string(self, **kwargs) -> str:
        """Kwargs is guaranteed to have all keys filled."""
        pass

    def get_string(self, **kwargs) -> None:
        self.validate_args(**kwargs)
        for arg, default_value in self.OPTIONAL_ARGS.items():
            if arg not in kwargs:
                kwargs[arg] = default_value
        string = self._get_string(**kwargs)
        print(string) # captured by the bash caller


# Add new string functions to STRING_FUNCTIONS to expose them to bash.

class ExampleExperimentName(StringFunction):
    NAME = "exp_name"
    REQUIRED_ARGS = ["dataset", "model"]
    OPTIONAL_ARGS = {"version": "v1", "batch_size": 32}

    def _get_string(self, **kwargs) -> str:
        return f"{kwargs['dataset']}_{kwargs['model']}_{kwargs['version']}_bs{kwargs['batch_size']}"



STRING_FUNCTIONS = [ExampleExperimentName]

ALL_STRING_FUNCTIONS = {func.NAME.lower(): func() for func in STRING_FUNCTIONS}


def parse():
    passed_in_args = sys.argv[1:]
    if len(passed_in_args) < 1:
        raise ValueError("Must provide at least the string name to get")
    string_name = passed_in_args[0].lower()
    if string_name not in ALL_STRING_FUNCTIONS:
        raise ValueError(f"String function {string_name} not found. Available string functions: {list(ALL_STRING_FUNCTIONS.keys())}")
    args = passed_in_args[1:]
    if len(args) % 2 != 0:
        raise ValueError(f"Arguments must be in the format --key value. Got: {args}")
    arg_dict = {}
    for i in range(0, len(args), 2):
        if not args[i].startswith("--"):
            raise ValueError(f"Argument keys must start with --. Got: {args[i]}")
        if args[i].strip("--") in arg_dict:
            raise ValueError(f"Duplicate argument key: {args[i][2:]}")
        if args[i+1].startswith("--"):
            raise ValueError(f"consecutive --s: {args}")
        if args[i + 1].strip().lower() == "none":
            arg_dict[args[i].strip("--").lower()] = None
        else:
            arg_dict[args[i].strip("--").lower()] = args[i + 1]
    return string_name, arg_dict


if __name__ == "__main__":
    string_name, args = parse()
    ALL_STRING_FUNCTIONS[string_name].get_string(**args)
