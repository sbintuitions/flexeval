from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class RewardBenchInstance:
    """A dataclass representing a triplet (prompt, chosen, rejected) of a
    reward bench task."""

    prompt: list[dict[str, str]]
    """
    The prompt for chosen/rejected responses.
    The format is a list of dictionaries, where each dictionary represents an OpenAI-format chat message,
    such as `{"role": "user", "content": "Hello!"}`.
    """
    chosen: list[dict[str, str]]
    """
    The chosen response to the prompt.
    The format is the same as `prompt`.
    """
    rejected: list[dict[str, str]]
    """
    The rejected response to the prompt.
    The format is the same as `prompt`.
    """
    category_key: str | None = None
    """
    A key to compute category-wise average accuracies.
    """
    extra_info: dict[str, Any] = field(default_factory=dict)
    """
    Extra information that can be used by passing to `Metric`.
    """


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
