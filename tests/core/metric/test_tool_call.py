from __future__ import annotations

import pytest

from flexeval.core.language_model.base import LMOutput
from flexeval.core.metric import MetricResult
from flexeval.core.metric.tool_call import ToolCallCount


@pytest.mark.parametrize(
    ("validation_results", "expected_ratios"),
    [
        (
            ["valid", "invalid", "valid", "valid"],
            {"tool_call_validation_result_ratio-valid": 3 / 4, "tool_call_validation_result_ratio-invalid": 1 / 4},
        ),
        (
            ["valid", "valid", "valid"],
            {"tool_call_validation_result_ratio-valid": 1.0},
        ),
        (
            [None, "valid", None],
            {"tool_call_validation_result_ratio-None": 2 / 3, "tool_call_validation_result_ratio-valid": 1 / 3},
        ),
        (
            [],
            {},
        ),
        (
            ["valid", "invalid", "parsing_error", "schema_error", "valid", "invalid"],
            {
                "tool_call_validation_result_ratio-valid": 2 / 6,
                "tool_call_validation_result_ratio-invalid": 2 / 6,
                "tool_call_validation_result_ratio-parsing_error": 1 / 6,
                "tool_call_validation_result_ratio-schema_error": 1 / 6,
            },
        ),
        (
            ["", "partial_valid", "timeout", "unknown_error"],
            {
                "tool_call_validation_result_ratio-": 0.25,
                "tool_call_validation_result_ratio-partial_valid": 0.25,
                "tool_call_validation_result_ratio-timeout": 0.25,
                "tool_call_validation_result_ratio-unknown_error": 0.25,
            },
        ),
    ],
)
def test_tool_call_count_ratios(validation_results: list[str], expected_ratios: dict[str, float]) -> None:
    """Test ToolCallCount ratio calculations with various validation result combinations."""
    metric = ToolCallCount()

    lm_outputs = [
        LMOutput(text=f"Response {i}", tool_call_validation_result=result)
        for i, result in enumerate(validation_results)
    ]
    references_list = [[f"ref{i}"] for i in range(len(validation_results))]

    result = metric.evaluate(lm_outputs, references_list)

    assert isinstance(result, MetricResult)
    assert result.summary == expected_ratios
    assert result.instance_details == [{"tool_call_validation_result": result} for result in validation_results]


@pytest.mark.parametrize(
    ("lm_outputs", "references_list"),
    [
        (["string output"], [["ref1"]]),
    ],
)
def test_tool_call_count_type_errors(lm_outputs: list[str], references_list: list[list[str]]) -> None:
    """Test that ToolCallCount raises TypeError for invalid input types."""
    metric = ToolCallCount()

    with pytest.raises(TypeError) as exc_info:
        metric.evaluate(lm_outputs, references_list)

    assert "ToolCallCount expects lm_outputs to be an LMOutput" in str(exc_info.value)
