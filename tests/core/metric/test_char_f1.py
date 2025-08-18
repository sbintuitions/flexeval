from __future__ import annotations

import pytest

from flexeval.core.metric import CharF1
from flexeval.core.metric.base import MetricResult
from flexeval.core.string_processor import AIONormalizer, RegexExtractor, StringProcessor


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "lm_output_processor", "reference_processor", "score"),
    [
        (["テスト"], [["テスト"]], None, None, 1.0),
        (["テスト"], [["テストです"]], None, None, 0.75),
        (["テスト"], [["テストです", "テスト"]], None, None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), RegexExtractor(r"\d+"), 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), None, 0.33),
        (
            ["答えは以下の通りです。\nA: 「蛹化（ようか）」"],
            [["蛹化"]],
            [RegexExtractor(r"A: (.*)"), AIONormalizer()],
            None,
            1.0,
        ),
    ],
    indirect=["lm_outputs"],
)
def test_char_f1(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    lm_output_processor: StringProcessor | list[StringProcessor] | None,
    reference_processor: StringProcessor | list[StringProcessor] | None,
    score: float,
) -> None:
    metric = CharF1(lm_output_processor=lm_output_processor, reference_processor=reference_processor)
    metric_result = metric.evaluate(lm_outputs, expected_outputs)
    assert metric_result.summary["char_f1"] == score
    assert metric_result.instance_details[0]["char_f1"] == score
    assert len(metric_result.instance_details) == len(lm_outputs)


def test_exact_match_with_category_key() -> None:
    """Test ExactMatch metric with category_key parameter."""
    metric = CharF1(category_key="category")

    extra_info_list = [
        {"category": "commonsense", "text": "This is sentence1."},
        {"category": "commonsense", "text": "This is sentence2."},
        {"category": "science", "text": "This is very scientific sentence."},
    ]
    lm_outputs = ["これは文1です。", "間違った訳", "これはすごく科学的な文です。"]
    references_list = [["これは文1です。"], ["これは文2です。"], ["これはすごく科学的な文です。"]]

    result = metric.evaluate(lm_outputs, references_list, extra_info_list)

    assert isinstance(result, MetricResult)
    assert "char_f1" in result.summary
    assert "char_f1/commonsense" in result.summary
    assert "char_f1/science" in result.summary

    assert pytest.approx(result.summary["char_f1"]) == 2 / 3
    assert pytest.approx(result.summary["char_f1/commonsense"]) == 1 / 2
    assert pytest.approx(result.summary["char_f1/science"]) == 1.0
