import pytest

from flexeval.core.reward_bench_dataset import RewardBenchInstance
from flexeval.core.reward_model.sequence_classification import SequenceClassificationRewardModel


@pytest.fixture(scope="module")
def reward_model() -> SequenceClassificationRewardModel:
    return SequenceClassificationRewardModel("tests/dummy_modules/llama-seq-classification-tiny")


def test_batch_judge(reward_model: SequenceClassificationRewardModel) -> None:
    instances = [
        RewardBenchInstance(
            prompt=[{"role": "user", "content": "Hello!"}],
            chosen=[{"role": "assistant", "content": "Hi!"}],
            rejected=[{"role": "assistant", "content": "Colorless green sleeps furiously."}],
            extra_info={"id": 1},
        ),
        RewardBenchInstance(
            prompt=[{"role": "user", "content": "Hello! How are you?"}],
            chosen=[{"role": "assistant", "content": "Hi! I am good."}],
            rejected=[{"role": "assistant", "content": "Colorless green sleeps furiously."}],
            extra_info={"id": 2},
        ),
    ]

    chosen_is_better, outputs = reward_model.batch_judge(instances)

    assert len(chosen_is_better) == len(instances)
    assert len(outputs) == len(instances)
    assert outputs[0].keys() == {"chosen_reward", "rejected_reward"}
