from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flexeval.core.reward_bench_dataset.hf import RewardBenchInstance


class RewardModel(ABC):
    """Judge which model is better given two items.

    The output is a tuple of the winner and the rationale.
    """

    @abstractmethod
    def batch_judge(
        self,
        batch_reward_bench_instances: list[RewardBenchInstance],
        gen_kwargs: dict[str, Any],
    ) -> list[str]:
        """Judge which model is better given a batch of item pairs.

        Args:
            batch_model_items: A list of tuples, each containing two model items.
        """
