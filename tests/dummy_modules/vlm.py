from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from flexeval.core.language_model import LanguageModel
from flexeval.core.language_model.base import LMOutput


def _decode_base64_image(data_url: str) -> Image.Image:
    _, base64_string = data_url.split(",")
    return Image.open(io.BytesIO(base64.b64decode(base64_string))).convert("RGB")


class DummyVLM(LanguageModel):
    """A dummy vision-language model for testing.

    It decodes every `image_url` content part (base64 data URL or file path)
    to verify the images are well-formed, then reports what it received,
    e.g. "2 image(s) and 1 text part(s) are given!".
    """

    def set_random_seed(self, seed: int) -> None:
        pass

    def _batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, Any]]],
        tools_list: list[list[dict[str, Any]] | None] | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        outputs = []
        for chat_messages in chat_messages_list:
            num_texts = 0
            images: list[Image.Image] = []
            for message in chat_messages:
                content = message["content"]
                if isinstance(content, str):
                    num_texts += 1
                    continue
                for part in content:
                    if part["type"] == "text":
                        num_texts += 1
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            images.append(_decode_base64_image(url))
                        else:
                            path = url.removeprefix("file://")
                            images.append(Image.open(path).convert("RGB"))
            outputs.append(
                LMOutput(
                    text=f"{len(images)} image(s) and {num_texts} text part(s) are given!",
                    finish_reason="stop",
                ),
            )
        return outputs
