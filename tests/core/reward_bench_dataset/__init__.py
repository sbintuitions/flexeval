from flexeval.core.reward_bench_dataset.hf import HFRewardBenchDataset


def test_load_hf_reward_bench_dataset() -> None:
    dataset = HFRewardBenchDataset(path="allenai/reward-bench", split="filtered")
    assert len(dataset) == 2985

    reward_bench_item = dataset[0]
    assert reward_bench_item.prompt == "How do I detail a car?"
