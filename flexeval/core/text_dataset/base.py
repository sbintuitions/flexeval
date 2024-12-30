from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class TextInstance:
    text: str
    prefix: str = ""


class TextDataset(Sequence[TextInstance], ABC):
    """
    This class represents a dataset of text examples.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> TextInstance:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"
