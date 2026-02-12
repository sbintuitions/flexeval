from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Literal, Protocol

from loguru import logger

from flexeval.core.chat_dataset.base import Preprocessor


class Image(Protocol):
    # PIL image
    def resize(self, size: tuple[int, int], resample: int) -> Image: ...
    def convert(self, mode: str) -> Image: ...
    def save(self, fp: BytesIO, format: str) -> None: ...  # noqa: A002


class ConvertImageToBase64(Preprocessor):
    """
    Preprocessor to convert image to base64 string.

    Args:
        key: The key in the input data that contains the image.
        image_format: The image format to use for encoding. Either "png" or "jpeg".
        max_length: The maximum length of the base64 string. Images will be resized
            to fit within this length if specified.
    """

    def __init__(self, key: str, image_format: Literal["png", "jpeg"] = "png", max_length: int | None = None) -> None:
        self.key = key
        self.image_format = image_format
        self.max_length = max_length

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        image = item[self.key]
        if image is None:
            base64_image = None
        elif isinstance(image, Image):
            base64_image = self.encode_image_to_base64(image, image_format=self.format, max_length=self.max_length)
        else:
            raise NotImplementedError("Unsupported image type: " + str(type(image)))

        item[f"{self.key}_base64"] = base64_image
        return item

    @staticmethod
    def encode_image_to_base64(
        image: Image,
        image_format: Literal["png", "jpeg"],
        max_length: int | None,
    ) -> str:
        def to_base64(img: Image) -> str:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format=image_format.upper())
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        base64_image = to_base64(image)
        if max_length is not None and len(base64_image) > max_length:
            factor = 1
            while len(base64_image) > max_length:
                factor *= 0.9
                new_size = (int(image.width * factor), int(image.height * factor))
                resized_image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
                base64_image = to_base64(resized_image)
            logger.debug(f"Image size reduced to {new_size} to fit max_length {max_length}.")
        return f"data:image/{image_format.lower()};base64,{base64_image}"
