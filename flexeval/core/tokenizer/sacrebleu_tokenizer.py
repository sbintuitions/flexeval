from __future__ import annotations

from sacrebleu.metrics.bleu import _get_tokenizer

from .base import Tokenizer


class SacreBleuTokenizer(Tokenizer):
    """
    A tokenizer imported from uses the sacrebleu library.

    Args:
        name: The name of the tokenizer.
    """

    def __init__(self, name: str) -> None:
        self.tokenizer = _get_tokenizer(name)()

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer(text).split(" ")
