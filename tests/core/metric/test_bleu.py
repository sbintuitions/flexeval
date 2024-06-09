from __future__ import annotations

import pytest

from flexeval.core.metric import BLEU


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["これはテストです"], [["これはテストです"]], 1.0),
        (
            ["これはテストです", "あれもテストです"],
            [["これはテストです", "これもテストでした"], ["あれもテストです", "あれはテスト"]],
            1.0,
        ),
        (["こんにちは、世界！"], [["こんばんわ, 地方？"]], 0.0),
        ([""], [["empty"]], 0.0),
    ],
)
def test_bleu(lm_outputs: list[str], expected_outputs: list[list[str]], score: float) -> None:
    bleu = BLEU(tokenize_option="ja-mecab")
    metric_result = bleu.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)
    assert metric_result.summary["bleu_score"] == pytest.approx(score)
    assert metric_result.instance_details[0]["bleu_score"] == pytest.approx(score)
