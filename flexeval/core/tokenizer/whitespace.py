from __future__ import annotations

from .base import Tokenizer


class WhitespaceTokenizer(Tokenizer):
    """
    A simple whitespace tokenizer.
    """

    def tokenize(self, text: str) -> list[str]:
        return text.split()
