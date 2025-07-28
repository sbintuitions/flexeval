from __future__ import annotations

import pytest

from flexeval.core.metric import MathVerify


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["The answer is $a+b$."], [["$b + a$ is the answer."]], 1.0),
        (["The answer is \\boxed{a+b}."], [["$b + a$ is the answer."]], 1.0),
        (["A: $x^2-1$."], [["The ratio is $(x-1)(x+1)$"]], 1.0),
        (["The answer is \\boxed{12}"], [["12"]], 1.0),
        (["The answer is \\boxed{1.5}"], [["1.5"]], 1.0),
        (["The answer is \\boxed{1.501}"], [["1.5"]], 0.0),
        (["答えは４です"], [["4"]], 0.0),
    ],
)
def test_exact_match(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    score: float,
) -> None:
    metric = MathVerify()
    metric_result = metric.evaluate(lm_outputs, references_list=expected_outputs)
    assert metric_result.summary["math_verify_accuracy"] == score
    assert isinstance(metric_result.instance_details[0]["math_verify_match"], int)
    assert isinstance(metric_result.instance_details[0]["extracted_answer"], list)
    assert len(metric_result.instance_details) == len(lm_outputs)
