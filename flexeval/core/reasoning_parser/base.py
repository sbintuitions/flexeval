from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Reasoning:
    text: str | None
    reasoning_text: str | None


class ReasoningParser(ABC):
    """Base class for parsing raw LLM output text into reasoning and content parts.

    Subclasses must implement `__call__`, which receives the raw text and returns
    an instance of `Reasoning` class.
    """

    @abstractmethod
    def __call__(self, raw_text: str) -> Reasoning:
        """Parse raw LLM output and return the reasoning and content parts.

        Args:
            raw_text: Raw output string produced by the language model.

        Returns:
            An instance of `Reasoning` class.
        """
        raise NotImplementedError
