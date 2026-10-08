"""
Deduplicated frame storage: each distinct image once, as lossless WebP in one uncompressed zip.
"""
import hashlib
import io
import os
import zipfile
from typing import Optional
import numpy as np
from PIL import Image
from cusi.agents.records import EncodedImage

FRAMES_FILE = "frames.zip"
WEBP_METHOD = 2


class FrameStore:
    """Writes <directory>/<name> and returns references "<name>/<k>.webp"; close() moves it into place."""

    def __init__(self, *, directory: str, name: str = FRAMES_FILE) -> None:
        self.name = name
        self.path = os.path.join(directory, name)
        self._tmp = self.path + ".tmp"
        self._zip = zipfile.ZipFile(self._tmp, "w", compression=zipfile.ZIP_STORED)
        self._refs: dict = {}

    def __call__(self, image) -> Optional[str]:
        if image is None:
            return None
        encoded = EncodedImage.of(image.pil() if isinstance(image, FrameRef) else image)
        key = hashlib.sha1(encoded.png).hexdigest()
        if key not in self._refs:
            member = f"{len(self._refs)}.webp"
            buf = io.BytesIO()
            encoded.pil().save(buf, format="WEBP", lossless=True, quality=100, method=WEBP_METHOD)
            self._zip.writestr(member, buf.getvalue())
            self._refs[key] = f"{self.name}/{member}"
        return self._refs[key]

    def close(self) -> None:
        self._zip.close()
        os.replace(self._tmp, self.path)


def read_frame(*, path: str) -> Image.Image:
    """The image at <dir>/<zip>/<member>, e.g. os.path.join(episode_dir, reference)."""
    archive, member = path.rsplit("/", 1)
    with zipfile.ZipFile(archive) as z:
        return Image.open(io.BytesIO(z.read(member))).convert("RGB")


class FrameRef:
    """A frame already written by a FrameStore, read on demand (pil()/numpy(), as EncodedImage)."""

    __slots__ = ("path",)

    def __init__(self, *, path: str) -> None:
        self.path = path

    def pil(self) -> Image.Image:
        return read_frame(path=self.path)

    def numpy(self) -> np.ndarray:
        return np.asarray(self.pil(), dtype=np.uint8)
