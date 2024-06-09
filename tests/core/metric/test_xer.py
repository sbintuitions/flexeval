from __future__ import annotations

import pytest

from flexeval.core.metric import XER
from flexeval.core.metric.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "cer_score", "wer_score"),
    [
        (["これは テスト"], [["これは テスト"]], 0.0, 0.0),
        (["こんにちは 世界"], [["こんばんわ 地方"]], 0.62, 1.0),
    ],
)
def test_rouge(lm_outputs: list[str], expected_outputs: list[list[str]], cer_score: float, wer_score: float) -> None:
    rouge = XER(tokenizer=WhitespaceTokenizer())
    metric_result = rouge.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)

    key = "cer_score"
    assert key in metric_result.summary
    assert isinstance(metric_result.summary[key], float)
    assert metric_result.summary[key] == pytest.approx(cer_score, abs=0.05)

    key = "wer_score"
    assert key in metric_result.summary
    assert isinstance(metric_result.summary[key], float)
    assert metric_result.summary[key] == pytest.approx(wer_score, abs=0.05)
