from __future__ import annotations

import pytest

from flexeval.core.metric import ExactMatch
from flexeval.core.metric.string_extractor import AIONormalizer, NoopNormalizer, Normalizer, RegexExtractor


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "normalizer", "reference_normalizer", "score"),
    [
        (["テスト"], [["テスト"]], None, None, 1.0),
        (["テスト"], [["テストです"]], None, None, 0.0),
        (["テスト"], [["テストです", "テスト"]], None, None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexExtractor(r"\d+"), NoopNormalizer(), 0.0),
        (
            ["答えは以下の通りです。\nA: 「蛹化（ようか）」"],
            [["蛹化"]],
            [RegexExtractor(r"A: (.*)"), AIONormalizer()],
            NoopNormalizer(),
            1.0,
        ),
    ],
)
def test_exact_match(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    normalizer: Normalizer | list[Normalizer] | None,
    reference_normalizer: Normalizer | list[Normalizer] | None,
    score: float,
) -> None:
    metric = ExactMatch(normalizer=normalizer, reference_normalizer=reference_normalizer)
    metric_result = metric.evaluate(lm_outputs, references_list=expected_outputs)
    assert metric_result.summary["exact_match"] == score
    assert isinstance(metric_result.instance_details[0]["exact_match"], int)
    assert len(metric_result.instance_details) == len(lm_outputs)
