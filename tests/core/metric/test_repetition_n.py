import pytest

from flexeval.core.metric.repetition_n import RepetitionN
from flexeval.core.tokenizer import WhitespaceTokenizer


def test_repetition_n_basic() -> None:
    metric = RepetitionN(n=2, tokenizer=WhitespaceTokenizer())
    outputs = ["a b c a b"]
    result = metric.evaluate(outputs, [[]], None)
    assert pytest.approx(result.summary["repetition_2"], 0.01) == 0.25
    assert pytest.approx(result.instance_details[0]["repetition_2"], 0.01) == 0.25


def test_repetition_n_empty() -> None:
    metric = RepetitionN(n=2, tokenizer=WhitespaceTokenizer())
    outputs = [""]
    result = metric.evaluate(outputs, [[]], None)
    assert result.summary["repetition_2"] == 0.0
    assert result.instance_details[0]["repetition_2"] == 0.0


def test_repetition_n_all_unique() -> None:
    metric = RepetitionN(n=2, tokenizer=WhitespaceTokenizer())
    outputs = ["a b c d"]
    result = metric.evaluate(outputs, [[]], None)
    assert result.summary["repetition_2"] == 0.0
    assert result.instance_details[0]["repetition_2"] == 0.0


def test_repetition_n_all_same() -> None:
    metric = RepetitionN(n=1, tokenizer=WhitespaceTokenizer())
    outputs = ["a a a a"]
    result = metric.evaluate(outputs, [[]], None)
    assert result.summary["repetition_1"] == 0.75
    assert result.instance_details[0]["repetition_1"] == 0.75
