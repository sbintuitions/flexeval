from __future__ import annotations

import pytest

from flexeval.core.language_model.base import LMOutput
from flexeval.core.metric import MetricResult
from flexeval.core.metric.finish_reason import FinishReasonCount


@pytest.mark.parametrize(
    ("finish_reasons", "expected_ratios", "expected_summary_len"),
    [
        (
            ["stop", "length", "stop", "stop"],
            {"finish_reason_ratio-stop": 3 / 4, "finish_reason_ratio-length": 1 / 4},
            2,
        ),
        (
            ["stop", "stop", "stop"],
            {"finish_reason_ratio-stop": 1.0},
            1,
        ),
        (
            [None, "stop", None],
            {"finish_reason_ratio-None": 2 / 3, "finish_reason_ratio-stop": 1 / 3},
            2,
        ),
        (
            ["stop", "length", "timeout", "error", "stop", "length"],
            {
                "finish_reason_ratio-stop": 2 / 6,
                "finish_reason_ratio-length": 2 / 6,
                "finish_reason_ratio-timeout": 1 / 6,
                "finish_reason_ratio-error": 1 / 6,
            },
            4,
        ),
        ([], {}, 0),
    ],
)
def test_finish_reason_count_functionality(
    finish_reasons: list[str], expected_ratios: list[dict[str, float]], expected_summary_len: int
) -> None:
    """Test FinishReasonCount metric functionality with various finish reason combinations."""
    metric = FinishReasonCount()

    lm_outputs = [LMOutput(text=f"Response {i+1}", finish_reason=reason) for i, reason in enumerate(finish_reasons)]
    references_list = [[f"ref{i+1}"] for i in range(len(finish_reasons))]

    result = metric.evaluate(lm_outputs, references_list)

    assert isinstance(result, MetricResult)
    for key, expected_value in expected_ratios.items():
        assert result.summary[key] == expected_value
    assert len(result.summary) == expected_summary_len
    assert result.instance_details == [{"finish_reason": reason} for reason in finish_reasons]


def test_finish_reason_count_empty_list() -> None:
    """Test FinishReasonCount with empty input lists."""
    metric = FinishReasonCount()

    lm_outputs = []
    references_list = []

    result = metric.evaluate(lm_outputs, references_list)

    assert isinstance(result, MetricResult)
    assert result.summary == {}
    assert result.instance_details == []


@pytest.mark.parametrize(
    ("lm_outputs", "references_list"),
    [
        (["string output"], [["ref1"]]),
    ],
)
def test_finish_reason_count_type_errors(lm_outputs: list[LMOutput], references_list: list[list[str]]) -> None:
    """Test that FinishReasonCount raises TypeError for invalid input types."""
    metric = FinishReasonCount()

    with pytest.raises(TypeError) as exc_info:
        metric.evaluate(lm_outputs, references_list)

    assert "FinishReasonMetric expects lm_outputs to be an LMOutput" in str(exc_info.value)
