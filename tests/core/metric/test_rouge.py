from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from flexeval.core.metric import ROUGE
from flexeval.core.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["これは テスト"], [["これは テスト"]], 1.0),
        (["こんにちは 世界"], [["こんばんわ 地方"]], 0.0),
        ([""], [["empty"]], 0.0),
        (["."], [["empty"]], 0.0),
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
        (["こんにちは 世界 です"], [["こんばんわ 地方"]], 2, 0.0),  # No match even after truncation
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


def test_rouge_with_recursion_limit() -> None:
    """Test that recursion limit is properly set and restored."""
    original_limit = sys.getrecursionlimit()
    custom_limit = 10000

    rouge = ROUGE(tokenizer=WhitespaceTokenizer(), recursion_limit=custom_limit)
    lm_outputs = ["これは テスト"]
    references_list = [["これは テスト"]]

    metric_result = rouge.evaluate(lm_outputs=lm_outputs, references_list=references_list)

    # Verify that recursion limit is restored after evaluation
    assert sys.getrecursionlimit() == original_limit

    # Verify basic functionality
    for key in ["rouge1", "rouge2", "rougeL"]:
        assert key in metric_result.summary
        assert isinstance(metric_result.summary[key], float)


def test_rouge_recursion_limit_is_set() -> None:
    """Test that sys.setrecursionlimit is called with the specified limit."""
    custom_limit = 10000

    with (
        patch("flexeval.core.metric.utils.sys.setrecursionlimit") as mock_setrecursionlimit,
        patch("flexeval.core.metric.utils.sys.getrecursionlimit", return_value=3000),
    ):
        rouge = ROUGE(tokenizer=WhitespaceTokenizer(), recursion_limit=custom_limit)
        rouge.evaluate(lm_outputs=["test"], references_list=[["test"]])

        # Verify setrecursionlimit was called with custom_limit
        assert mock_setrecursionlimit.call_count == 2
        mock_setrecursionlimit.assert_any_call(custom_limit)
        # Verify it was restored to original value
        mock_setrecursionlimit.assert_any_call(3000)
