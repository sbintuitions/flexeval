from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Sequence

import datasets


@dataclass
class RewardBenchInstance:
    """A dataclass representing a triplet (prompt, chosen, rejected) of a
    reward bench task."""

    prompt: str
    chosen: str
    rejected: str


class RewardBenchDataset(Sequence[RewardBenchInstance], ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of instances in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int) -> RewardBenchInstance:
        """Returns the i-th instance."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"

