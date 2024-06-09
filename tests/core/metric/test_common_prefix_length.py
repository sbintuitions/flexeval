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


def test_common_prefix_length() -> None:
    metric = CommonPrefixLength()

    lm_outputs = ["これはペンです", "これはペンギンです"]
    references_list = [["これはペンです"], ["これはペンです", "これはペンギンです"]]

    metric_result = metric.evaluate(lm_outputs, references_list=references_list)

    assert metric_result.summary == {
        "average_common_prefix_length": 8.0,  # average of len("これはペンです")=7 and len("これはペンギンです")=9
        "longest_common_prefix_length": 9,  # the length of "これはペンギンです"
    }
    assert metric_result.instance_details == [{"common_prefix_length": 7}, {"common_prefix_length": 9}]
