from __future__ import annotations

import pytest

from flexeval.core.metric.common_string_length import (
    CommonStringLength,
    get_longest_common_substring,
)


@pytest.mark.parametrize(
    ("s1", "s2", "common_substring"),
    [
        ("", "", ""),
        ("", "a", ""),
        ("a", "", ""),
        ("a", "a", "a"),
        ("abc", "ab", "ab"),
        ("Yes, これはペンです", "これはペンギンです", "これはペン"),
    ],
)
def test_get_longest_common_substring(s1: str, s2: str, common_substring: str) -> None:
    assert get_longest_common_substring(s1, s2) == common_substring


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
def test_common_string_length(
    lm_outputs: list[str], references_list: list[list[str]], common_prefix_lengths: list[int]
) -> None:
    metric = CommonStringLength()

    metric_result = metric.evaluate(lm_outputs, references_list=references_list)
    assert metric_result.summary == {
        "average_common_string_length": sum(common_prefix_lengths) / len(common_prefix_lengths),
        "longest_common_string_length": max(common_prefix_lengths),
    }
    assert metric_result.instance_details == [{"common_string_length": length} for length in common_prefix_lengths]
