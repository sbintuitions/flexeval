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


def test_common_string_length() -> None:
    metric = CommonStringLength()

    lm_outputs = ["これはペンです", "これはペンギンです"]
    references_list = [["これはペンです"], ["これはペンです", "これはペンギンです"]]

    metric_result = metric.evaluate(lm_outputs, references_list=references_list)
    assert metric_result.summary == {
        "average_common_string_length": 8.0,  # average of len("これはペンです")=7 and len("これはペンギンです")=9
        "longest_common_string_length": 9,  # the length of "これはペンギンです"
    }
    assert metric_result.instance_details == [{"common_string_length": 7}, {"common_string_length": 9}]
