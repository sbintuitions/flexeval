from __future__ import annotations

import pytest

from flexeval.core.metric import CharF1
from flexeval.core.metric.string_processor import AIONormalizer, NoopNormalizer, RegexExtractor, StringProcessor


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "processor", "reference_processor", "score"),
    [
        (["テスト"], [["テスト"]], None, None, 1.0),
        (["テスト"], [["テストです"]], None, None, 0.75),
        (["テスト"], [["テストです", "テスト"]], None, None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), NoopNormalizer(), 0.33),
        (
            ["答えは以下の通りです。\nA: 「蛹化（ようか）」"],
            [["蛹化"]],
            [RegexExtractor(r"A: (.*)"), AIONormalizer()],
            NoopNormalizer(),
            1.0,
        ),
    ],
)
def test_char_f1(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    processor: StringProcessor | list[StringProcessor] | None,
    reference_processor: StringProcessor | list[StringProcessor] | None,
    score: float,
) -> None:
    metric = CharF1(processor=processor, reference_processor=reference_processor)
    metric_result = metric.evaluate(lm_outputs, expected_outputs)
    assert metric_result.summary["char_f1"] == score
    assert metric_result.instance_details[0]["char_f1"] == score
    assert len(metric_result.instance_details) == len(lm_outputs)
