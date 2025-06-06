from __future__ import annotations

import pytest

from flexeval.core.metric import BLEU
from flexeval.core.metric.base import MetricResult
from flexeval.core.string_processor.base import StringProcessor
from flexeval.core.string_processor.regex import RegexExtractor
from flexeval.core.string_processor.string_strip import StringStrip


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "lm_output_processor", "reference_processor", "score"),
    [
        (["これはテストです"], [["これはテストです"]], None, None, 1.0),
        (
            ["これはテストです", "あれもテストです"],
            [["これはテストです", "これもテストでした"], ["あれもテストです", "あれはテスト"]],
            None,
            None,
            1.0,
        ),
        (["こんにちは、世界！"], [["こんばんわ, 地方？"]], None, None, 0.0),
        ([""], [["empty"]], None, None, 0.0),
        (["訳: これはテストです"], [["これはテストです "]], RegexExtractor(r"訳:\s*(.+)"), StringStrip(), 1.0),
    ],
)
def test_bleu(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    lm_output_processor: StringProcessor | list[StringProcessor] | None,
    reference_processor: StringProcessor | list[StringProcessor] | None,
    score: float,
) -> None:
    bleu = BLEU(
        tokenize_option="ja-mecab", lm_output_processor=lm_output_processor, reference_processor=reference_processor
    )
    metric_result = bleu.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)
    assert metric_result.summary["bleu_score"] == pytest.approx(score)
    assert metric_result.instance_details[0]["bleu_score"] == pytest.approx(score)
    assert len(metric_result.instance_details) == len(lm_outputs)


def test_exact_match_with_category_key() -> None:
    """Test ExactMatch metric with category_key parameter."""
    metric_w_cat_key = BLEU(tokenize_option="ja-mecab", category_key="category")
    metric_wo_cat_key = BLEU(tokenize_option="ja-mecab")
    extra_info_list = [
        {"category": "commonsense", "text": "This is sentence1."},
        {"category": "commonsense", "text": "This is sentence2."},
        {"category": "science", "text": "This is very scientific sentence."},
    ]
    lm_outputs = ["これは文1です。", "間違った訳", "これはすごく科学的な文です。"]
    references_list = [["これは文1です。"], ["これは文2です。"], ["これはすごく科学的な文です。"]]

    result_w_cat_key = metric_w_cat_key.evaluate(lm_outputs, references_list, extra_info_list)
    result_wo_cat_key = metric_wo_cat_key.evaluate(lm_outputs, references_list, extra_info_list)

    assert isinstance(result_w_cat_key, MetricResult)
    assert "bleu_score" in result_w_cat_key.summary
    assert "sentence_bleu_score/commonsense" in result_w_cat_key.summary
    assert "sentence_bleu_score/science" in result_w_cat_key.summary

    # Confirm that the overall bleu_score do not change with or without the category key
    assert pytest.approx(result_w_cat_key.summary["bleu_score"]) == result_wo_cat_key.summary["bleu_score"]

    assert pytest.approx(result_w_cat_key.summary["sentence_bleu_score/commonsense"]) == 0.5
    assert pytest.approx(result_w_cat_key.summary["sentence_bleu_score/science"]) == 1.0
