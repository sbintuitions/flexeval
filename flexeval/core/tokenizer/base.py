from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Tokenizer interface.

    Tokenizers are used to split text into tokens.
    Typically, this is used in `Metric` that requires word-level statistics.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError
