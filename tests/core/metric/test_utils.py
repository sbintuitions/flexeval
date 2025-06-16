from __future__ import annotations

from typing import TypeVar

import pytest

from flexeval.core.metric.utils import aggregate_category_wise_scores, apply_string_processors, validate_inputs
from flexeval.core.string_processor.lower import StringLower
from flexeval.core.string_processor.string_strip import StringStrip

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


def test_apply_string_processors() -> None:
    text = "  Hello WORLD  "

    # Test with None processors
    result = apply_string_processors(text, None)
    assert result == text

    # Test with single processor
    lower_processor = StringLower()
    result = apply_string_processors(text, lower_processor)
    assert result == "  hello world  "

    # Test with multiple processors
    strip_processor = StringStrip()
    processors = [strip_processor, lower_processor]
    result = apply_string_processors(text, processors)
    assert result == "hello world"


def test_validate_inputs() -> None:
    lm_outputs = ["output1", "output2"]
    references_list = [["ref1"], ["ref2"]]
    extra_info_list = [{"key": "value1"}, {"key": "value2"}]

    # Test valid inputs
    validate_inputs(lm_outputs, references_list, extra_info_list)
    validate_inputs(lm_outputs, references_list, None)

    # Test length mismatch between lm_outputs and references_list
    with pytest.raises(ValueError) as exc_info:
        validate_inputs(["output1"], [["ref1"], ["ref2"]], None)
    assert "Number of model outputs (1) and number of references (2)" in str(exc_info.value)

    # Test length mismatch between extra_info_list and lm_outputs
    with pytest.raises(ValueError) as exc_info:
        validate_inputs(lm_outputs, references_list, [{"key": "value1"}])
    assert "Number of extra_info entries (1) should match number of outputs (2)" in str(exc_info.value)
