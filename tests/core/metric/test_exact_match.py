from __future__ import annotations

import pytest

from flexeval.core.metric import ExactMatch


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["テスト"], [["テスト"]], 1.0),
        (["テスト"], [["テストです"]], 0.0),
    ],
)
def test_exact_match(lm_outputs: list[str], expected_outputs: list[list[str]], score: float) -> None:
    metric = ExactMatch()
    metric_result = metric.evaluate(lm_outputs, references_list=expected_outputs)
    assert metric_result.summary["exact_match"] == score
    assert isinstance(metric_result.instance_details[0]["exact_match"], int)
