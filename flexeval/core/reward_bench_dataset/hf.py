from __future__ import annotations

from typing import Any

import datasets

from flexeval.core.reward_bench_dataset.base import RewardBenchDataset, RewardBenchInstance


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
