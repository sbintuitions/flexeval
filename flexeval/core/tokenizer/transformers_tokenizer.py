from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from .base import Tokenizer


class TransformersTokenizer(Tokenizer):
    def __init__(
        self,
        path: str,
        init_kwargs: dict[str, Any] | None = None,
        tokenize_kwargs: dict[str, Any] | None = None,
    ) -> None:
        init_kwargs = init_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(path, **init_kwargs)
        self.tokenize_kwargs = tokenize_kwargs or {}

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text, **self.tokenize_kwargs)
