from __future__ import annotations

from flexeval.core.reward_bench_dataset.hf import RewardBenchDataset, RewardBenchInstance


class DummyRewardBenchDataset(RewardBenchDataset):
    def __init__(self) -> None:
        self.items = [
            RewardBenchInstance(
                prompt=f"prompt_text_{i}",
                chosen=f"chosen_text_{i}",
                rejected=f"rejected_text_{i}",
            )
            for i in range(100)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> RewardBenchDataset:
        return self.items[i]
