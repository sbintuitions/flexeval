from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class GenerationInstance:
    """
    A dataclass representing a single input-output pair of a generation task.
    """

    inputs: dict[str, str]
    """
    Inputs of the generation task.
    This will be embedded into the prompt for the language model in `PromptTemplate`.
    """
    references: list[str]
    """
    Reference outputs for the generation task.
    The model's output will be evaluated against these references in `Metric`.
    """


class GenerationDataset(Sequence[GenerationInstance], ABC):
    """A dataset holding `GenerationInstance`."""

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of instances in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int) -> GenerationInstance:
        """
        Returns the i-th instance.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"
