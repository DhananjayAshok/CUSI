import os
import yaml
import datetime
from typing import Any, Optional
from cusi.utils.fundamental import meta_dict_to_str
from cusi.utils.log_handling import log_warn
import hashlib


def hash_meta_dict(meta_dict: dict[str, Any]) -> str:
    """10-character SHA-256 hex hash of the dict's canonical string."""
    meta_str = meta_dict_to_str(meta_dict)
    hash_obj = hashlib.sha256(meta_str.encode("utf-8"))
    hex_hash = hash_obj.hexdigest()[
        :10
    ]
    return hex_hash


def write_meta(
    write_dir: str,
    meta_dict: dict[str, Any],
    parameters: Optional[dict[str, Any]] = None,
) -> str:
    """Write ``meta_dict`` (plus a write_timestamp) to ``write_dir/meta_{hash}.yaml`` and return the hash."""
    meta_hash = hash_meta_dict(meta_dict)
    meta_dict["write_timestamp"] = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    meta_path = os.path.join(write_dir, f"meta_{meta_hash}.yaml")
    if not os.path.exists(write_dir):
        log_warn(
            f"Directory {write_dir} does not exist for meta file writing. Creating one, but it is recommended that you do it yourself...",
            parameters=parameters,
        )
        os.makedirs(write_dir)
    with open(meta_path, "w") as f:
        yaml.dump(meta_dict, f)
    return meta_hash


def add_meta_details(
    meta_dict: dict[str, Any], add_details: dict[str, Any]
) -> dict[str, Any]:
    """Copy of ``meta_dict`` updated with ``add_details``; the original is not modified."""
    alt = meta_dict.copy()
    alt.update(add_details)
    return alt
