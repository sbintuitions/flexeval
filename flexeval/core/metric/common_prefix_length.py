from __future__ import annotations

from .base import Metric, MetricResult


def get_longest_common_prefix(s1: str, s2: str) -> str:
    """Find the longest common prefix between two strings."""
    length = min(len(s1), len(s2))

    for i in range(length):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:length]


class CommonPrefixLength(Metric):
    """
    A metric that calculates the length of the longest common prefix between the model output and the reference.

    Examples:
        >>> from flexeval import CommonPrefixLength
        >>> common_prefix_length = CommonPrefixLength()
        >>> lm_outputs = ["ABCDEFG"]
        >>> references_list = [["ABCdefg"]]
        >>> result = common_prefix_length.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={"average_common_prefix_length": 3.0, "longest_common_prefix_length": 3},
            instance_details=[{"common_prefix_length": 3}],
        )
    """

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        common_prefix_length_list: list[int] = []
        for lm_output, references in zip(lm_outputs, references_list):
            common_prefix_length = max(len(get_longest_common_prefix(lm_output, gt)) for gt in references)
            common_prefix_length_list.append(common_prefix_length)

        return MetricResult(
            {
                "average_common_prefix_length": sum(common_prefix_length_list) / len(common_prefix_length_list),
                "longest_common_prefix_length": max(common_prefix_length_list),
            },
            instance_details=[{"common_prefix_length": s} for s in common_prefix_length_list],
        )
