from __future__ import annotations

import pytest

from flexeval.core.metric import XER
from flexeval.core.tokenizer import SacreBleuTokenizer, WhitespaceTokenizer


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "cer_score", "wer_score"),
    [
        (["これは テスト"], [["これは テスト"]], 0.0, 0.0),
        (["こんにちは 世界"], [["こんばんわ 地方"]], 0.62, 1.0),
    ],
    indirect=["lm_outputs"],
)
def test_rouge(lm_outputs: list[str], expected_outputs: list[list[str]], cer_score: float, wer_score: float) -> None:
    rouge = XER(tokenizer=WhitespaceTokenizer())
    metric_result = rouge.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)

    assert len(metric_result.instance_details) == len(lm_outputs)

    key = "cer_score"
    assert key in metric_result.summary
    assert isinstance(metric_result.summary[key], float)
    assert metric_result.summary[key] == pytest.approx(cer_score, abs=0.05)

    key = "wer_score"
    assert key in metric_result.summary
    assert isinstance(metric_result.summary[key], float)
    assert metric_result.summary[key] == pytest.approx(wer_score, abs=0.05)


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs"),
    [
        (["これは テスト"], [["これは テスト"]]),
        (["こんにちは 世界"], [["こんばんわ 地方"]]),
    ],
    indirect=["lm_outputs"],
)
def test_instance_wer_score_matches_summary_with_tokenizer(
    lm_outputs: list[str], expected_outputs: list[list[str]]
) -> None:
    # With a single instance, the summary wer_score (computed over tokenized text) must equal
    # instance_details[0]["wer_score"], since the corpus-level WER over one pair is the same
    # ratio as the instance-level WER for that pair.
    xer = XER(tokenizer=WhitespaceTokenizer())
    metric_result = xer.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)

    assert metric_result.instance_details[0]["wer_score"] == pytest.approx(metric_result.summary["wer_score"])


def test_instance_wer_score_for_unsegmented_language() -> None:
    # Japanese has no whitespace between words, so wer() on the raw, untokenized text treats
    # each entire sentence as a single "word": any mismatch forces the instance-level wer_score
    # to exactly 1.0, regardless of how similar the sentences actually are.
    # "今日は晴天です" and "今日は雨天です" tokenize (via MeCab) to 4 tokens each, differing only in
    # one token ("晴天"/"雨天"), so the correct token-level WER is 1/4 = 0.25 -- neither 0.0 nor 1.0.
    lm_outputs = ["今日は晴天です"]
    expected_outputs = [["今日は雨天です"]]

    xer = XER(tokenizer=SacreBleuTokenizer(name="ja-mecab"))
    metric_result = xer.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)

    instance_wer_score = metric_result.instance_details[0]["wer_score"]
    assert instance_wer_score == pytest.approx(metric_result.summary["wer_score"])
    assert instance_wer_score == pytest.approx(0.25)
