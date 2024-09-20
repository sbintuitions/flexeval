from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flexeval.core.reward_bench_dataset.hf import RewardBenchInstance


class RewardModel(ABC):
    """Base class for reward models."""

    @abstractmethod
    def batch_judge(
        self,
        batch_reward_bench_instances: list[RewardBenchInstance],
    ) -> tuple[list[Any], list[bool]]:
        """Judge a batch of reward bench instances.

        Args:
            batch_reward_bench_instances (list[RewardBenchInstance]): A list of tuples, each containing two model items.

        Returns:
            tuple[list[Any], list[bool]]: A tuple of the judge outputs and the chosen_is_betters.
        """
