from __future__ import annotations

from flexeval.core.reward_bench_dataset import RewardBenchDataset, RewardBenchInstance


class DummyRewardBenchDataset(RewardBenchDataset):
    def __init__(self) -> None:
        self.items = [
            RewardBenchInstance(
                prompt=[{"role": "user", "content": f"prompt_text_{i}"}],
                chosen=[{"role": "user", "content": f"chosen_text_{i}"}],
                rejected=[{"role": "user", "content": f"rejected_text_{i}"}],
                extra_info={"id": i},
            )
            for i in range(100)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> RewardBenchInstance:
        return self.items[i]
