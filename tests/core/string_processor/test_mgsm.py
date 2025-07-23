from __future__ import annotations

import pytest

from flexeval.core.metric.exact_match import ExactMatch
from flexeval.core.string_processor import RemoveCommaProcessor, SimpleEvalMGSMProcessor

lm_output_processor = SimpleEvalMGSMProcessor()
reference_processor = RemoveCommaProcessor()
numerical_expression_metric = ExactMatch(
    lm_output_processor=lm_output_processor,
    reference_processor=reference_processor,
)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("He needs to remove 15 toys. The final Answer: 15", "15"),
        (r"t = 7.5 \times 2 = 15. There fore, I need to remove \(\boxed{15}\) toys.", "15"),  # with tailing period.
        ("The final Answer: 1000.", "1000"),  # with tailing period.
        ("The final Answer: 1000.00", "1000"),  # with tailing zeros.
        ("The final Answer: 0.314", "0.314"),  # decimal
        ("The final Answer: -5", "-5"),  # negative number
        ("The final Answer: -3.1419", "-3.1419"),  # negative decimal
        ("The final Answer: 1,319", "1319"),  # with comma
        (r"The final Answer: $\boxed{540 \, \text{meters}}$.", "540"),  # with confusing comma.
        (r"The final Answer: \boxed{230 + x}", "230"),
    ],
)
def test_lm_output_processor(before: str, after: str) -> None:
    assert lm_output_processor(before) == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("1,000", "1000"),
    ],
)
def test_reference_processor(before: str, after: str) -> None:
    assert reference_processor(before) == after


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["12.0"], [["12"]], 1),
    ],
)
def test_numerical_expression_metric(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    score: int,
) -> None:
    metric_result = numerical_expression_metric.evaluate(lm_outputs, references_list=expected_outputs)
    assert metric_result.summary["exact_match"] == score
