from __future__ import annotations

import pytest

from flexeval.core.metric.common_prefix_length import (
    CommonPrefixLength,
    get_longest_common_prefix,
)


@pytest.mark.parametrize(
    ("s1", "s2", "common_prefix"),
    [
        ("", "", ""),
        ("", "a", ""),
        ("a", "", ""),
        ("a", "a", "a"),
        ("これはペンです", "これはペンギンです", "これはペン"),
        ("Yes, これはペンです", "これはペンギンです", ""),
    ],
)
def test_get_longest_common_prefix(s1: str, s2: str, common_prefix: str) -> None:
    assert get_longest_common_prefix(s1, s2) == common_prefix


@pytest.mark.parametrize(
    ("lm_outputs", "references_list", "common_prefix_lengths"),
    [
        (
            ["これはペンです", "これはペンギンです"],
            [["これはペンです"], ["これはペンです", "これはペンギンです"]],
            [7, 9],
        ),
    ],
    indirect=["lm_outputs"],
)
def test_common_prefix_length(
    lm_outputs: list[str], references_list: list[list[str]], common_prefix_lengths: list[int]
) -> None:
    metric = CommonPrefixLength()

    metric_result = metric.evaluate(lm_outputs=lm_outputs, references_list=references_list)

    assert metric_result.summary == {
        "average_common_prefix_length": sum(common_prefix_lengths) / len(common_prefix_lengths),
        "longest_common_prefix_length": max(common_prefix_lengths),
    }
    assert metric_result.instance_details == [{"common_prefix_length": length} for length in common_prefix_lengths]
