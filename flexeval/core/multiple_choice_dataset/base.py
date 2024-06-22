from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class MultipleChoiceInstance:
    """
    A dataclass representing a single input-output pair of a multiple-choice task.
    """

    inputs: dict[str, str]
    """
    Inputs of the multiple-choice task.
    This will be embedded into the prompt for the language model in `PromptTemplate`.
    """
    choices: list[str]
    """
    Choices for the multiple-choice task.
    `LanguageModel` will choose the answer based on the log-probabilities of these choices.
    """
    answer_index: int
    """
    Index of the correct answer in `choices`.
    """


class MultipleChoiceDataset(Sequence[MultipleChoiceInstance], ABC):
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of instances in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int) -> MultipleChoiceInstance:
        """
        Returns the i-th instance.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"
