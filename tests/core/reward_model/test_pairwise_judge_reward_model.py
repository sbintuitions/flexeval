import pytest

from flexeval.core.reward_model.pairwise_judge_reward_model import PairwiseChoice, evaluate_model_output


@pytest.mark.parametrize(
    "model_output,gold_label,expected",
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
def test_evaluate_model_output(model_output, gold_label, expected):
    result = evaluate_model_output(model_output, gold_label)
    assert result is expected
