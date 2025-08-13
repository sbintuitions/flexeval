import pytest

from flexeval import Jinja2PromptTemplate
from flexeval.core.reward_bench_dataset import RewardBenchInstance
from flexeval.core.reward_model.pairwise_judge_reward_model import (
    PairwiseChoice,
    PairwiseJudgeRewardModel,
    aggregate_judge_results,
    evaluate_model_output,
)
from tests.dummy_modules import DummyRewardLanguageModel


@pytest.mark.parametrize(
    ("model_output", "gold_label", "expected"),
    [
        # contains only correct label --> OK
        ("This answer is [[A]] because it's better.", PairwiseChoice.A, True),
        ("I pick [[B]] since it’s more informative.", PairwiseChoice.B, True),
        # contains both labels --> NG
        ("I think both [[A]] and [[B]] have merit.", PairwiseChoice.A, False),
        ("This is tricky. [[A]] and also [[B]]", PairwiseChoice.B, False),
        # contains incorrect labels --> NG
        ("The answer is clearly [[B]]", PairwiseChoice.A, False),
        ("My choice is [[A]]", PairwiseChoice.B, False),
        # contains neither labels --> NG
        ("I cannot decide between the options.", PairwiseChoice.A, False),
        ("Neither seems particularly good.", PairwiseChoice.B, False),
    ],
)
def test_evaluate_model_output(model_output: str, gold_label: PairwiseChoice, expected: bool) -> None:
    result = evaluate_model_output(model_output, gold_label)
    assert result is expected


@pytest.mark.parametrize(("num_samples"), [1, 10, 100])
def test_pairwise_judge_reward_model(num_samples: int):
    reward_model = PairwiseJudgeRewardModel(
        language_model=DummyRewardLanguageModel(),
        prompt_template=Jinja2PromptTemplate(template="{{ prompt }}\t{{ answer_a }}\t{{ answer_b }}"),
    )

    instances = [
        RewardBenchInstance(
            prompt=[{"role": "user", "content": "How is the weather today"}],
            chosen=[{"role": "assistant", "content": "Sunny"}],
            rejected=[{"role": "assistant", "content": "I don't know"}],
        )
    ] * num_samples

    final_results, outputs = reward_model.batch_judge(instances)
    assert len(final_results) == num_samples
    assert len(outputs) == num_samples
    assert not any(final_results), "All evaluation results should be False"


class _MockOut:
    """dummy judge_output element with text field"""

    def __init__(self, text: str):
        self.text = text


@pytest.mark.parametrize(
    ("pairs", "expected_finals", "expected_consistencies"),
    [
        (
            [(True, True), (True, False)],
            [True, False],
            [True, False],
        ),
        (
            [(False, False), (False, True)],
            [False, False],
            [True, False],
        ),
    ],
)
def test_aggregate_multiple_instances(pairs: list, expected_finals: list, expected_consistencies: list):
    n = len(pairs)
    outputs = [{"llm_inputs": [f"ab_{i}", f"ba_{i}"]} for i in range(n)]

    judge_outputs = []
    chosen_is_better_list = []
    for i, (ab_eval, ba_eval) in enumerate(pairs):
        judge_outputs.append(_MockOut(f"ab_text_{i}"))
        judge_outputs.append(_MockOut(f"ba_text_{i}"))
        chosen_is_better_list.extend([ab_eval, ba_eval])

    final_results, final_outputs = aggregate_judge_results(outputs, judge_outputs, chosen_is_better_list)

    assert final_results == expected_finals
    for i in range(n):
        assert final_outputs[i]["consistent"] == expected_consistencies[i]
        assert final_outputs[i]["final_is_correct"] == expected_finals[i]
        assert final_outputs[i]["llm_outputs"] == [f"ab_text_{i}", f"ba_text_{i}"]
        assert final_outputs[i]["evaluation_results"] == list(pairs[i])
