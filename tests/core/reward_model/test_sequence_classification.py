import pytest

from flexeval.core.reward_bench_dataset import RewardBenchInstance
from flexeval.core.reward_model.sequence_classification import SequenceClassificationRewardModel


@pytest.fixture(scope="module")
def reward_model() -> SequenceClassificationRewardModel:
    return SequenceClassificationRewardModel("tests/dummy_modules/llama-seq-classification-tiny")


def test_batch_judge(reward_model: SequenceClassificationRewardModel) -> None:
    num_instances = 4
    instances = [
        RewardBenchInstance(
            prompt=[{"role": "user", "content": "Hello!"}],
            chosen=[{"role": "assistant", "content": "Hi!"}],
            rejected=[{"role": "assistant", "content": "Hello!"}],
            extra_info={"id": i},
        )
        for i in range(num_instances)
    ]

    chosen_is_better, outputs = reward_model.batch_judge(instances)

    assert len(chosen_is_better) == num_instances
    assert len(outputs) == num_instances
    assert outputs[0].keys() == {"chosen_reward", "rejected_reward"}
