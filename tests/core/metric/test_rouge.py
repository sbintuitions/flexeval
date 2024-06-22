from __future__ import annotations

import pytest

from flexeval.core.metric import ROUGE
from flexeval.core.metric.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["これは テスト"], [["これは テスト"]], 1.0),
        (["こんにちは 世界"], [["こんばんわ 地方"]], 0.0),
        ([""], [["empty"]], 0.0),
    ],
)
def test_rouge(lm_outputs: list[str], expected_outputs: list[list[str]], score: float) -> None:
    rouge = ROUGE(tokenizer=WhitespaceTokenizer())
    metric_result = rouge.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)

    for key in ["rouge1", "rouge2", "rougeL"]:
        assert key in metric_result.summary
        assert isinstance(metric_result.summary[key], float)
        assert metric_result.summary[key] == pytest.approx(score)
        assert metric_result.instance_details[0][key] == pytest.approx(score)
        assert len(metric_result.instance_details) == len(lm_outputs)
