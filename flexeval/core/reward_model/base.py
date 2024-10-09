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
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        """Judge a batch of reward bench instances.

        Args:
            batch_reward_bench_instances (list[RewardBenchInstance]): A list of tuples, each containing two model items.

        Returns:
            tuple[list[bool], list[Any]]: A tuple with the following elements:
                - chosen_is_betters: Indicating whether each `chosen` item is considered better by the model.
                - judge_outputs: A list of outputs (rationale, score, etc....) from the model.
        """
