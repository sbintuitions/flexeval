from __future__ import annotations

import pytest

from flexeval.core.metric import CharF1
from flexeval.core.metric.normalizer import AIONormalizer, NoopNormalizer, Normalizer, RegexNormalizer


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "normalizer", "reference_normalizer", "score"),
    [
        (["テスト"], [["テスト"]], None, None, 1.0),
        (["テスト"], [["テストです"]], None, None, 0.75),
        (["テスト"], [["テストです", "テスト"]], None, None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexNormalizer(r"\d+"), None, 1.0),
        (["The answer is 10."], [["Answer: 10"]], RegexNormalizer(r"\d+"), NoopNormalizer(), 0.33),
        (
            ["答えは以下の通りです。\nA: 「蛹化（ようか）」"],
            [["蛹化"]],
            [RegexNormalizer(r"A: (.*)"), AIONormalizer()],
            NoopNormalizer(),
            1.0,
        ),
    ],
)
def test_char_f1(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    normalizer: Normalizer | list[Normalizer] | None,
    reference_normalizer: Normalizer | list[Normalizer] | None,
    score: float,
) -> None:
    metric = CharF1(normalizer=normalizer, reference_normalizer=reference_normalizer)
    metric_result = metric.evaluate(lm_outputs, expected_outputs)
    assert metric_result.summary["char_f1"] == score
    assert metric_result.instance_details[0]["char_f1"] == score
    assert len(metric_result.instance_details) == len(lm_outputs)
