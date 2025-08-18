from __future__ import annotations

import pytest

from flexeval.core.metric import OutputLengthStats


@pytest.mark.parametrize(
    ("lm_outputs", "lm_output_lengths"),
    [
        (["123456"], [6]),
        (["123456", "123456789"], [6, 9]),
    ],
    indirect=["lm_outputs"],
)
def test_output_length_stats(lm_outputs: list[str], lm_output_lengths: list[int]) -> None:
    metric = OutputLengthStats()
    metric_result = metric.evaluate(lm_outputs=lm_outputs, references_list=[])

    expected_summary = {
        "avg_output_length": sum(lm_output_lengths) / len(lm_output_lengths),
        "max_output_length": max(lm_output_lengths),
        "min_output_length": min(lm_output_lengths),
    }
    assert metric_result.summary == pytest.approx(expected_summary)
    assert metric_result.instance_details[0]["output_length"] == lm_output_lengths[0]
    assert len(metric_result.instance_details) == len(lm_outputs)
