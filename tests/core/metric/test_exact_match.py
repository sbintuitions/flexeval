from __future__ import annotations

import pytest

from flexeval.core.metric import ExactMatch
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
    assert metric_result.summary["exact_match"] == score
    assert isinstance(metric_result.instance_details[0]["exact_match"], int)
    assert len(metric_result.instance_details) == len(lm_outputs)
