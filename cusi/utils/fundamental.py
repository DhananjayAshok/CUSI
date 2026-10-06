# Fundamental utilities that do not rely on any other file.
import os
import logging
from typing import Any


def get_logger(
    level: int = logging.INFO, filename: str = None, add_console: bool = True
) -> logging.Logger:
    """Logger to the console and/or ``filename`` (None means no file logging)."""
    fmt_str = "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt_str)
    logger = logging.getLogger("PROJECT_NAME")
    if add_console:
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt_str)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a")
        log_formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
        logger.propagate = False
    return logger


def meta_dict_to_str(
    meta_dict: dict[str, Any],
    *,
    print_mode: bool = False,
    n_indents: int = 1,
    skip_write_timestamp: bool = True,
) -> str:
    """Indented multi-line string in print mode; otherwise a compact sorted key-value string for hashing."""
    keys = list(meta_dict.keys())
    keys.sort()
    meta_str = ""
    for key in keys:
        if print_mode:
            indent = "\t" * n_indents
            meta_str += f"{indent}{key}: {meta_dict[key]}\n"
        else:
            if skip_write_timestamp and key == "write_timestamp":
                continue
            meta_str += f"{key.lower().strip()}_{str(meta_dict[key]).lower().strip()}"
    return meta_str


def logger_print_dict(
    logger: logging.Logger, meta_dict: dict[str, Any], n_indents: int = 1
) -> None:
    meta_dict_str = meta_dict_to_str(
        meta_dict, print_mode=True, n_indents=n_indents, skip_write_timestamp=False
    )
    logger.info(meta_dict_str)


def file_makedir(file_path: str) -> None:
    """Create the parent directories of ``file_path`` if missing."""
    dirname = os.path.dirname(file_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
    return
