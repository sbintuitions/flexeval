from __future__ import annotations

import pytest

from flexeval import Correlation, MetricResult


@pytest.mark.parametrize(
    ("method", "lm_outputs", "references", "expected_correlation"),
    [
        ("pearson", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
        ("pearson", [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0),
        ("spearman", [1, 2, 3, 4, 5], [1, 20, 30, 400, 500], 1.0),
        ("spearman", [1, 2, 3, 4, 5], [500, 400, 30, 20, 1], -1.0),
        ("kendall", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
        ("kendall", [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0),
    ],
)
def test_correlation(
    method: str, lm_outputs: list[float], references: list[float], expected_correlation: float
) -> None:
    correlation = Correlation(method=method)
    references_list = [[ref] for ref in references]  # Wrap references in a list for each instance

    result = correlation.evaluate(lm_outputs, references_list)

    assert isinstance(result, MetricResult)
    assert f"{method}_correlation" in result.summary
    assert result.summary[f"{method}_correlation"] == pytest.approx(expected_correlation, rel=1e-3)


def test_instantiation_fails_with_invalid_method() -> None:
    with pytest.raises(ValueError, match="Invalid method"):  # Expecting an error for invalid method
        Correlation(method="invalid")


def test_evaluation_fails_with_mismatched_lengths() -> None:
    correlation = Correlation(method="pearson")

    lm_outputs = [1, 2, 3]
    references_list = [[1], [2]]  # Mismatched lengths

    with pytest.raises(ValueError):
        correlation.evaluate(lm_outputs, references_list)


def test_evaluation_does_not_fail_with_non_numeric_lm_outputs() -> None:
    correlation = Correlation(method="pearson")

    lm_outputs = ["1", "a", "3"]
    references_list = [["1.0"], ["2.0"], ["3.0"]]

    with pytest.warns(UserWarning, match="Failed to convert model output 'a' to float"):
        result = correlation.evaluate(lm_outputs, references_list)

    assert result.summary["pearson_correlation"] is not None


def test_evaluation_fails_with_non_numeric_references() -> None:
    correlation = Correlation(method="pearson")

    lm_outputs = ["1", "2", "3"]
    references_list = [["1.0"], ["non-numeric"], ["3.0"]]

    with pytest.raises(ValueError):
        correlation.evaluate(lm_outputs, references_list)
