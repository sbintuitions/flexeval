from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import random
from typing import Any, Literal, Sequence

import datasets


@dataclass
class RewardBenchInstance:
    """A dataclass representing a triplet (prompt, chosen, rejected) of a
    reward bench task."""

    prompt: str
    chosen: str
    rejected: str

class Choice(str, Enum):
    A = "[[A]]"
    B = "[[B]]"


@dataclass
class ShufflePairwiseInstance:
    prompt: str
    answer_a: str
    answer_b: str
    chosen: Literal[Choice.A, Choice.B]


def reward_bench_instance_to_shuffle_pairwise_instance(
    reward_bench_instance: RewardBenchInstance,
) -> ShufflePairwiseInstance:
    # Choose a random number between 0 and 1
    random_choice = random.randint(0, 1)
    if random_choice == 0:
        answer_a, answer_b, chosen = reward_bench_instance.chosen, reward_bench_instance.rejected, Choice.A
    else:
        answer_a, answer_b, chosen = reward_bench_instance.rejected, reward_bench_instance.chosen, Choice.B
    return ShufflePairwiseInstance(
        prompt=reward_bench_instance.prompt,
        answer_a=answer_a,
        answer_b=answer_b,
        chosen=chosen,
    )

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


class HFRewardBenchDataset(RewardBenchDataset):
    """Load RewardBenchInstance from a huggingface dataset.

    Args:
        path: The name or path of the huggingface dataset.
        split: The split of the dataset to use.
        subset: The subset of the dataset to use.
        dataset_kwargs: The keyword arguments for loading the dataset.
    """

    def __init__(
        self,
        path: str,
        split: str,
        subset: str | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        dataset_kwargs = dataset_kwargs or {}
        items = datasets.load_dataset(path, split=split, name=subset, **dataset_kwargs)
        self.items = [dict(item) for item in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> RewardBenchInstance:
        item = self.items[i]
        return RewardBenchInstance(
            prompt=item["prompt"],
            chosen=item["chosen"],
            rejected=item["rejected"],
        )
