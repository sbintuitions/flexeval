from __future__ import annotations

import pytest

from flexeval.core.metric import OutputLengthStats


@pytest.mark.parametrize(
    ("lm_outputs", "expected_summary"),
    [
        (["123456"], {"avg_output_length": 6, "max_output_length": 6, "min_output_length": 6}),
        (["123456", "123456789"], {"avg_output_length": 7.5, "max_output_length": 9, "min_output_length": 6}),
    ],
)
def test_output_length_stats(lm_outputs: list[str], expected_summary: dict[str, float]) -> None:
    metric = OutputLengthStats()
    metric_result = metric.evaluate(lm_outputs=lm_outputs, references_list=[])
    assert metric_result.summary == pytest.approx(expected_summary)
    assert metric_result.instance_details[0]["output_length"] == len(lm_outputs[0])
    assert len(metric_result.instance_details) == len(lm_outputs)
