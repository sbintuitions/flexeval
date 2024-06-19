from __future__ import annotations

import pytest

from flexeval.core.metric import CharF1


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["テスト"], [["テスト"]], 1.0),
        (["テスト"], [["テストです"]], 0.75),
        (["テスト"], [["テストです", "テスト"]], 1.0),
        (["abc"], [["cba"]], 0.33),
        (["abc"], [["def"]], 0.0),
    ],
)
def test_exact_match(lm_outputs: list[str], expected_outputs: list[list[str]], score: float) -> None:
    metric = CharF1()
    metric_result = metric.evaluate(lm_outputs, expected_outputs)
    assert metric_result.summary["char_f1"] == score
    assert metric_result.instance_details[0]["char_f1"] == score
    assert len(metric_result.instance_details) == len(lm_outputs)
