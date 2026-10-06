import os
import yaml
from typing import Any, Optional
from cusi.utils.fundamental import get_logger


def load_yaml(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def compute_secondary_parameters(params: dict[str, Any]) -> None:
    """Derive (and create) directories, log_file, logger and vLLM_base_url in-place from the base config."""
    params["data_dir"] = os.path.join(params["storage_dir"], "data")
    params["model_dir"] = os.path.join(params["storage_dir"], "models")
    params["tmp_dir"] = os.path.join(params["storage_dir"], "tmp")
    params["log_dir"] = os.path.join(params["results_dir"], "logs")
    params["figure_dir"] = os.path.join(params["results_dir"], "figures")
    params["vLLM_base_url"] = f"http://localhost:{params['vllm_port']}/v1/"
    for dirname in [
        "data_dir",
        "model_dir",
        "log_dir",
        "figure_dir",
        "tmp_dir",
    ]:
        if not os.path.exists(params[dirname]):
            os.makedirs(params[dirname])
    if "log_file" not in params:
        log_file = os.path.join(params["log_dir"], "log.txt")
        params["log_file"] = log_file
    else:
        # normalise // so the log_dir prefix check is reliable
        log_dir_str = params["log_dir"].replace("//", "/")
        log_file_str = params["log_file"].replace("//", "/")
        if not log_file_str.startswith(log_dir_str):
            log_file = os.path.join(params["log_dir"], params["log_file"])
            params["log_file"] = log_file
    logger = get_logger(filename=params["log_file"])
    params["logger"] = logger


def load_parameters(parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Merge every configs/*.yaml into the parameters dict; an already-loaded dict is returned as is."""
    if parameters is not None:
        if (
            "logger" not in parameters
        ):  # this is a flag that secondary parameters need to be computed
            compute_secondary_parameters(parameters)
        return parameters
    essential_keys = ["storage_dir", "results_dir"]
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cusi/utils/ -> root
    params = {"project_root": project_root}
    logger = get_logger()
    config_files = os.listdir(os.path.join(project_root, "configs"))

    def error(msg):
        logger.error(msg)
        raise ValueError(msg)

    if "private_vars.yaml" not in config_files:
        error("Please create private_vars.yaml in the configs directory")
    for file in config_files:
        if file.endswith(".yaml"):
            configs = load_yaml(os.path.join(project_root, "configs", file))
            for key in configs:
                if key in params:
                    error(
                        f"{key} is present in multiple config files. At least one of which is {file}. Please remove the duplicate"
                    )
            params.update(configs)
        else:
            pass

    for key in params:
        if params[key] == "PLACEHOLDER":
            error(
                f"{key} is currently the placeholder value in private_vars.yaml. Please set it"
            )
    for essential_key in essential_keys:
        if essential_key not in params:
            error(f"Please set {essential_key} in one of the config yamls")
    if os.path.exists(params["storage_dir"]):
        if any([f.endswith(".py") for f in os.listdir(params["storage_dir"])]):
            logger.warning(
                f"There are .py files in the storage_dir {params['storage_dir']}. It is recommended to set a path which has nothing else inside it to avoid issues."
            )
    else:
        os.makedirs(params["storage_dir"])
        logger.info(f"Created storage directory {params['storage_dir']}")
    compute_secondary_parameters(params)
    return params
