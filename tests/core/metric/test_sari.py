from __future__ import annotations

import pytest

from flexeval.core.metric.sari import SARI


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "extra_info_list", "expected_score"),
    [
        (
            ["About 95 you now get in."],
            [
                [
                    "About 95 species are currently known.",
                    "About 95 species are now accepted.",
                    "95 species are now accepted.",
                ]
            ],
            [{"source": "About 95 species are currently accepted."}],
            0.2695,  #  from huggingface/evaluate https://huggingface.co/spaces/evaluate-metric/sari/blob/main/sari.py
        ),
        (
            ["Cat on mat."],
            [["The cat sat on the mat.", "The cat is on the mat.", "The cat sat."]],
            [{"source": "The cat perched on the mat."}],
            0.3131,  #  from huggingface/evaluate https://huggingface.co/spaces/evaluate-metric/sari/blob/main/sari.py
        ),
        (
            ["This sample is for a test."],
            [["This sample is for a test."]],
            [{"source": "This instance is for trial run."}],
            1.0,
        ),
        (
            ["1 2 3 4"],
            [["a b c d"]],
            [{"source": "a b c d"}],
            0.0,
        ),
    ],
    indirect=["lm_outputs"],
)
def test_sari(
    lm_outputs: list[str],
    expected_outputs: list[list[str]],
    extra_info_list: list[dict[str, str]],
    expected_score: float,
) -> None:
    sari_flexeval = SARI(source_key="source")
    metric_result = sari_flexeval.evaluate(
        lm_outputs=lm_outputs, references_list=expected_outputs, extra_info_list=extra_info_list
    )

    assert pytest.approx(metric_result.summary["sari_score"], abs=1e-3) == expected_score
    assert all(name in metric_result.summary for name in ["sari_score", "sari_add", "sari_keep", "sari_del"])
    assert len(metric_result.instance_details) == len(lm_outputs)
