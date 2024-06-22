from __future__ import annotations

from .base import Metric, MetricResult


def get_longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring between two strings."""
    m = len(s1)
    n = len(s2)

    # Create a table to store lengths of longest common suffixes of substrings
    # dp[i][j] will be the length of longest common suffix of s1[0..i-1] and s2[0..j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # To store length of the longest common substring
    length = 0

    # To store the index of the cell which contains the maximum value
    # This cell's indices will be used to build up the answer
    end_index = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > length:
                    length = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0

    # If there is no common substring
    if length == 0:
        return ""

    # Return the longest common substring
    return s1[end_index - length : end_index]


class CommonStringLength(Metric):
    """
    A metric that calculates the length of the longest common substring between the model output and the reference.

    Examples:
        >>> from flexeval import CommonStringLength
        >>> common_string_length = CommonStringLength()
        >>> lm_outputs = ["aBCDEFG"]
        >>> references_list = [["ABCDefg"]]
        >>> result = common_string_length.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={"average_common_string_length": 3.0, "longest_common_string_length": 3},
            instance_details=[{"common_string_length": 3}],
        )
    """

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        common_string_length_list: list[int] = []
        for lm_output, references in zip(lm_outputs, references_list):
            common_string_length = max(len(get_longest_common_substring(lm_output, gt)) for gt in references)
            common_string_length_list.append(common_string_length)

        return MetricResult(
            {
                "average_common_string_length": sum(common_string_length_list) / len(common_string_length_list),
                "longest_common_string_length": max(common_string_length_list),
            },
            instance_details=[{"common_string_length": s} for s in common_string_length_list],
        )
