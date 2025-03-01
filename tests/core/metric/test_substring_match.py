from __future__ import annotations

import pytest

from flexeval import MetricResult, SubstringMatch


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["cat dog", "dog cat"], [["cat"], ["dog"]], 1.0),
        (["cat", "dog", "mouse"], [["cat"], ["dog"], ["elephant"]], 0.6667),
        (["", "cat dog"], [["anything"], ["cat"]], 0.5),
        (["Substring is not present"], [["missing"]], 0.0),
    ],
)
def test_substring_match_any_mode(lm_outputs: list[str], expected_outputs: list[list[str]], score: float) -> None:
    """Test SubstringMatch in 'any' mode with various inputs."""
    metric = SubstringMatch(mode="any")
    metric_result = metric.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)
    assert round(metric_result.summary["substring_match-any"], 4) == round(score, 4)
    assert all(isinstance(detail["substring_match"], bool) for detail in metric_result.instance_details)
    assert len(metric_result.instance_details) == len(lm_outputs)


@pytest.mark.parametrize(
    ("lm_outputs", "expected_outputs", "score"),
    [
        (["cat dog"], [["cat", "dog"]], 1.0),
        (["cat dog mouse"], [["cat", "dog"]], 1.0),
        (["cat"], [["cat", "dog"]], 0.0),
        (["cat dog", "dog cat", "elephant"], [["cat", "dog"], ["dog", "cat"], ["elephant", "trunk"]], 0.6667),
        (["The quick brown fox"], [["quick", "fox"]], 1.0),
        (["The quick brown fox"], [["quick", "zebra"]], 0.0),
        (["one two three"], [["one", "two", "three"]], 1.0),
        (["one three"], [["one", "two", "three"]], 0.0),
    ],
)
def test_substring_match_all_mode(lm_outputs: list[str], expected_outputs: list[list[str]], score: float) -> None:
    metric = SubstringMatch(mode="all")
    metric_result = metric.evaluate(lm_outputs=lm_outputs, references_list=expected_outputs)
    assert round(metric_result.summary["substring_match-all"], 4) == round(score, 4)
    assert all(isinstance(detail["substring_match"], bool) for detail in metric_result.instance_details)
    assert len(metric_result.instance_details) == len(lm_outputs)


def test_default_mode() -> None:
    """Test that the default mode is 'any'."""
    metric = SubstringMatch()
    assert metric.mode == "any"
    assert metric.match_func == any


def test_invalid_mode() -> None:
    """Test that an invalid mode raises a ValueError."""
    with pytest.raises(ValueError, match="mode must be 'any' or 'all'"):
        SubstringMatch(mode="invalid")


def test_mismatched_input_lengths() -> None:
    """Test that mismatched input lengths raise a ValueError."""
    metric = SubstringMatch()
    with pytest.raises(ValueError, match="lm_outputs and references_list must have the same length"):
        metric.evaluate(lm_outputs=["cat"], references_list=[["cat"], ["dog"]])


def test_empty_inputs() -> None:
    """Test with empty inputs."""
    metric = SubstringMatch()
    result = metric.evaluate(lm_outputs=[], references_list=[])
    assert isinstance(result, MetricResult)
    assert len(result.instance_details) == 0
