from __future__ import annotations

import pytest

from flexeval.core.metric.repetition_count import RepetitionCount


@pytest.mark.parametrize(
    ("lm_outputs", "count_threshold", "threshold_length", "expected_ratio"),
    [
        (["hello hello hello", "hello"], 3, 5, 0.5),  # 1 repetition in the first text
        (["hello hello hello", "hello"], 3, 10, 0.0),  # No repetition because of the increased threshold_length
        (["hello hello hello", "hello"], 10, 5, 0.0),  # No repetition because of the increased count_threshold
    ],
)
def test_get_most_repeated_pattern(
    lm_outputs: list[str], count_threshold: int, threshold_length: int, expected_ratio: float
) -> None:
    metric = RepetitionCount(count_threshold=count_threshold, threshold_length=threshold_length)
    result = metric.evaluate(lm_outputs, references_list=[[]] * len(lm_outputs))
    assert result.summary["repetition_ratio"] == expected_ratio
