from __future__ import annotations

import base64
import dataclasses
from io import BytesIO
from typing import Any, Literal

from loguru import logger
from PIL import Image

from .template_based import Preprocessor

Data = dict[str, Any]


def image_to_base64(image: Image.Image, image_format: str, max_length: int | None = None) -> str:
    """Encode a PIL image as a `data:image/...;base64,...` URL string.

    If `max_length` is given and the encoded string exceeds it, the image is
    iteratively downscaled until the encoded string fits.
    """

    def _to_base64(img: Image.Image) -> str:
        buffer = BytesIO()
        img.convert("RGB").save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    base64_image = _to_base64(image)
    if max_length is not None and len(base64_image) > max_length:
        factor = 1.0
        new_size = image.size
        while len(base64_image) > max_length:
            factor *= 0.9
            new_size = (int(image.width * factor), int(image.height * factor))
            resized_image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
            base64_image = _to_base64(resized_image)
        logger.debug(f"Image size reduced to {new_size} to fit max_length {max_length}.")
    return f"data:image/{image_format.lower()};base64,{base64_image}"


def load_image(image: Image.Image | dict) -> Image.Image:
    """Normalize an image value into a PIL Image.

    Accepts PIL Image objects or dicts of the form `{"bytes": ..., "path": ...}`
    (the raw form stored in parquet files that is obtained when images are
    loaded without the HF `Image` feature metadata, e.g. via the `parquet`
    dataset loader).
    """
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"]))
        if image.get("path") is not None:
            return Image.open(image["path"])
    msg = f"Unsupported image value: {type(image)}"
    raise TypeError(msg)


@dataclasses.dataclass
class ConvertImageToBase64(Preprocessor):
    """Convert an image stored under `key` to a base64 data-URL string stored
    under `{key}_base64`, ready to be embedded as an OpenAI-compatible
    `image_url` content part.
    """

    key: str
    format: Literal["PNG", "JPEG"] = "PNG"
    max_length: int | None = None

    def __call__(self, item: Data) -> Data:
        image = item[self.key]
        if image is None:
            base64_image = None
        else:
            base64_image = image_to_base64(
                load_image(image),
                image_format=self.format,
                max_length=self.max_length,
            )

        item[f"{self.key}_base64"] = base64_image
        return item


@dataclasses.dataclass
class ConvertImageListToBase64(Preprocessor):
    """Convert a list of images (e.g., for multi-image VQA) stored under `key`
    to a list of base64 data-URL strings stored under `{key}_base64`.
    """

    key: str
    format: Literal["PNG", "JPEG"] = "PNG"
    max_length: int | None = None

    def __call__(self, item: Data) -> Data:
        images = item[self.key]
        if images is None:
            item[f"{self.key}_base64"] = None
        elif isinstance(images, (list, tuple)):
            item[f"{self.key}_base64"] = [
                image_to_base64(load_image(image), image_format=self.format, max_length=self.max_length)
                for image in images
            ]
        else:
            msg = f"Unsupported image list type: {type(images)}"
            raise TypeError(msg)

        return item


@dataclasses.dataclass
class EnsureMinSize(Preprocessor):
    """Ensure the image under `key` is at least `min_size` pixels on its
    shorter side, upscaling in place (preserving aspect ratio) if necessary.
    """

    key: str
    min_size: int
    interpolation: Image.Resampling = Image.Resampling.BILINEAR

    def __call__(self, item: Data) -> Data:
        image = item[self.key]
        if image is None:
            return item

        if not isinstance(image, Image.Image):
            msg = f"Unsupported image type: {type(image)}"
            raise TypeError(msg)

        width, height = image.size
        if width >= self.min_size and height >= self.min_size:
            return item
        if width <= height:
            new_width = self.min_size
            new_height = int(self.min_size * height / width)
        else:
            new_height = self.min_size
            new_width = int(self.min_size * width / height)
        item[self.key] = image.resize((new_width, new_height), resample=self.interpolation)
        return item
