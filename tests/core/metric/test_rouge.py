from __future__ import annotations

import pytest

from flexeval.core.metric import ROUGE
from flexeval.core.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["これは テスト"], [["これは テスト"]], 1.0),
        (["こんにちは 世界"], [["こんばんわ 地方"]], 0.0),
        ([""], [["empty"]], 0.0),
    ],
    indirect=["lm_outputs"],
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


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "max_output_tokens", "score"),
    [
        (["これは テスト です"], [["これは テスト"]], 2, 1.0),  # Exact match after truncation
        (["こんにちは 世界 です"], [["こんばんわ 地方"]], 2, 0.0),  # No match after truncation
        (["同じ 単語 ではない"], [["同じ 単語"]], 2, 1.0),  # Match after truncation
    ],
    indirect=["lm_outputs"],
)
def test_rouge_with_max_output_tokens(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    max_output_tokens: int,
    score: float,
) -> None:
    rouge = ROUGE(tokenizer=WhitespaceTokenizer(), max_output_tokens=max_output_tokens)
    metric_result = rouge.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)

    for key in ["rouge1", "rouge2", "rougeL"]:
        assert key in metric_result.summary
        assert isinstance(metric_result.summary[key], float)
        assert metric_result.summary[key] == pytest.approx(score)
        assert metric_result.instance_details[0][key] == pytest.approx(score)
        assert len(metric_result.instance_details) == len(lm_outputs)
