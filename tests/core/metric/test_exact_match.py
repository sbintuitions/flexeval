from __future__ import annotations

import pytest

from flexeval.core.metric import ExactMatch, MetricResult
from flexeval.core.string_processor import AIONormalizer, RegexExtractor, StringProcessor


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "lm_output_processor", "reference_processor", "score"),
    [
        (["テスト"], [["テスト"]], None, None, 1.0),
        (["テスト"], [["テストです"]], None, None, 0.0),
        (["テスト"], [["テストです", "テスト"]], None, None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), RegexExtractor(r"\d+"), 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), None, 0.0),
        (
            ["答えは以下の通りです。\nA: 「蛹化（ようか）」"],
            [["蛹化"]],
            [RegexExtractor(r"A: (.*)"), AIONormalizer()],
            None,
            1.0,
        ),
    ],
)
def test_exact_match(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    lm_output_processor: StringProcessor | list[StringProcessor] | None,
    reference_processor: StringProcessor | list[StringProcessor] | None,
    score: float,
) -> None:
    metric = ExactMatch(lm_output_processor=lm_output_processor, reference_processor=reference_processor)
    metric_result = metric.evaluate(lm_outputs, references_list=expected_outputs)

    assert isinstance(metric_result, MetricResult)
    assert metric_result.summary["exact_match"] == score
    assert isinstance(metric_result.instance_details[0]["exact_match"], int)
    assert len(metric_result.instance_details) == len(lm_outputs)


def test_exact_match_with_category_key() -> None:
    """Test ExactMatch metric with category_key parameter."""
    metric = ExactMatch(category_key="category")

    task_inputs_list = [
        {"category": "binary", "text": "Is this true?"},
        {"category": "binary", "text": "Is this false?"},
        {"category": "binary", "text": "Is this correct?"},
        {"category": "open", "text": "What do you think?"},
    ]
    lm_outputs = ["yes", "no", "yes", "maybe"]
    references_list = [["yes"], ["no"], ["no"], ["maybe"]]

    result = metric.evaluate(lm_outputs, references_list, task_inputs_list)

    assert isinstance(result, MetricResult)
    assert "exact_match" in result.summary
    assert "exact_match/binary" in result.summary
    assert "exact_match/open" in result.summary

    # Overall accuracy: 3/4 = 0.75 (yes, no, maybe matched correctly)
    assert result.summary["exact_match"] == 0.75
    # Binary category accuracy: 2/3 = ~0.67 (2 correct out of 3)
    assert pytest.approx(result.summary["exact_match/binary"]) == 2 / 3
    # Open category accuracy: 1/1 = 1.0 (1 correct out of 1)
    assert result.summary["exact_match/open"] == 1.0

    # Check instance details
    assert len(result.instance_details) == 4
    assert result.instance_details[0]["exact_match"] is True  # "yes" matches
    assert result.instance_details[1]["exact_match"] is True  # "no" matches
    assert result.instance_details[2]["exact_match"] is False  # "yes" doesn't match "no"
    assert result.instance_details[3]["exact_match"] is True  # "maybe" matches
