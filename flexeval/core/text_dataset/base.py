from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class TextDataset(Sequence[str], ABC):
    """
    This class represents a dataset of text examples.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> str:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"
