from __future__ import annotations

from .base import Tokenizer


class MecabTokenizer(Tokenizer):
    """
    MeCab tokenizer for Japanese text.
    """

    def __init__(self) -> None:
        import fugashi

        self._tagger = fugashi.Tagger("-Owakati")

    def tokenize(self, text: str) -> list[str]:
        tokens = self._tagger(text)
        return [token.surface for token in tokens]
