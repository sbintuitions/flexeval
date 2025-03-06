from __future__ import annotations

from typing import TypeVar

import pytest

from flexeval.core.metric.utils import aggregate_category_wise_scores

T = TypeVar("T")


@pytest.mark.parametrize(
    ("scores", "categories", "expected", "error_msg"),
    [
        # Normal case with multiple categories
        (
            [1.0, 2.0, 3.0, 4.0],
            ["A", "A", "B", "B"],
            {"A": 1.5, "B": 3.5},
            None,
        ),
        # Single category case
        (
            [1.0, 2.0, 3.0],
            ["A", "A", "A"],
            {"A": 2.0},
            None,
        ),
        # Boolean scores case
        (
            [True, False, True],
            ["X", "X", "Y"],
            {"X": 0.5, "Y": 1.0},
            None,
        ),
        # Length mismatch error case
        (
            [1.0, 2.0],
            ["A", "B", "C"],
            None,
            "Length of scores (2) and category_keys (3) must be the same",
        ),
    ],
)
def test_aggregate_category_wise_scores(
    scores: list[float | bool], categories: list[T], expected: dict[T, float], error_msg: str | None
) -> None:
    if error_msg is not None:
        with pytest.raises(ValueError) as exc_info:
            aggregate_category_wise_scores(scores, categories)
        assert error_msg in str(exc_info.value)
    else:
        result = aggregate_category_wise_scores(scores, categories)
        assert result == expected
