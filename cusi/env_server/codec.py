"""
JSON encoding of what an env returns. uint8 images (H, W, 3 or 1) become {"__png__": base64}; other arrays
{"__array__": nested list, "dtype"}; objects with to_dict() {"__obj__": class name, **fields}; enums
{"__enum__": "module:Class.MEMBER"}; classes {"__class__": "module:Class"}; anything else str(). The client (user_template/envclient.py) decodes the same markers without importing CUSI.
"""
import base64
import enum
import importlib
import io
import numpy as np
from PIL import Image

MAX_DEPTH = 12


def _png(arr: np.ndarray) -> str:
    img = arr[:, :, 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def encode(value, _depth: int = 0):
    if _depth > MAX_DEPTH:      # a self-referencing structure (GameBoyRL's step records): give up on it
        return str(value)[:200]
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8 and (value.ndim == 2 or (value.ndim == 3 and value.shape[2] in (1, 3, 4))):
            return {"__png__": _png(value)}
        return {"__array__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, enum.Enum):
        return {"__enum__": f"{type(value).__module__}:{type(value).__qualname__}.{value.name}"}
    if isinstance(value, dict):
        return {(k.__name__ if isinstance(k, type) else str(k)): encode(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, _depth + 1) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, type):
        return {"__class__": f"{value.__module__}:{value.__qualname__}"}
    if hasattr(value, "to_dict"):
        return {"__obj__": type(value).__name__, **encode(value.to_dict(), _depth + 1)}
    return str(value)


def decode(value):
    """The inverse for CUSI-side consumers (remote_env.py): PNGs back to uint8 arrays, TextBox dicts to TextBoxes."""
    if isinstance(value, dict):
        if "__png__" in value:
            arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(value["__png__"]))))
            return arr[:, :, None] if arr.ndim == 2 else arr
        if "__array__" in value:
            return np.asarray(value["__array__"], dtype=value["dtype"])
        if "__enum__" in value or "__class__" in value:      # the same code base on both sides (CUSI's agents)
            module, name = (value.get("__enum__") or value["__class__"]).split(":")
            obj = importlib.import_module(module)
            for part in name.split("."):
                obj = getattr(obj, part)
            return obj
        if "__obj__" in value:
            fields = {k: decode(v) for k, v in value.items() if k != "__obj__"}
            if value["__obj__"] == "TextBox":
                from cusi.envs.text_boxes import TextBox
                return TextBox.from_dict(fields)
            return fields
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value
