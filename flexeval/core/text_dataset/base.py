from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class TextDataset(ABC):
    """
    This class represents a dataset of text examples.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        pass
