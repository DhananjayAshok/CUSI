from typing import Any, Optional
from cusi.utils.fundamental import meta_dict_to_str
from cusi.utils.parameter_handling import load_parameters


def log_error(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    """Log the error and raise RuntimeError; only for unrecoverable errors."""
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.error(message, stacklevel=2)
    raise RuntimeError()


def log_warn(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.warn(message, stacklevel=2)


def log_info(message: str, parameters: Optional[dict[str, Any]] = None) -> None:
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.info(message, stacklevel=2)


def log_dict(
    meta_dict: dict[str, Any],
    *,
    n_indents: int = 1,
    parameters: Optional[dict[str, Any]] = None
) -> None:
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    meta_dict_str = meta_dict_to_str(
        meta_dict, print_mode=True, n_indents=n_indents, skip_write_timestamp=False
    )
    logger.info(meta_dict_str, stacklevel=2)
