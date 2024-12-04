import pytest

from flexeval.core.language_model import HuggingFaceLM
from flexeval.core.reward_bench_dataset import RewardBenchInstance
from flexeval.core.reward_model import LogProbRewardModel


@pytest.fixture(scope="module")
def reward_model() -> LogProbRewardModel:
    hf_lm = HuggingFaceLM("sbintuitions/tiny-lm-chat")
    return LogProbRewardModel(hf_lm)


def test_batch_judge(reward_model: LogProbRewardModel) -> None:
    num_instances = 4
    instances = [
        RewardBenchInstance(
            prompt=[{"role": "user", "content": "Hello! How are you?"}],
            chosen=[{"role": "assistant", "content": "I'm good, thank you!"}],
            rejected=[{"role": "assistant", "content": "!?!??!?@#@#左様"}],
            extra_info={"id": i},
        )
        for i in range(num_instances)
    ]

    chosen_is_better, outputs = reward_model.batch_judge(instances)

    assert chosen_is_better == [True, True, True, True]
    assert len(outputs) == num_instances
    assert outputs[0].keys() == {"chosen_log_prob", "rejected_log_prob"}
