from __future__ import annotations

from typing import Literal

from .base import Metric, MetricResult


class SubstringMatch(Metric):
    """
    A metric that calculates how many outputs contain any of the expected substrings.

    Args:
        mode: The mode to calculate the substring match.
            - "any": If any of the expected substrings are in the output, it is a match.
            - "all": If all of the expected substrings are in the output, it is a match.

    Examples:
        >>> from flexeval import SubstringMatch
        >>> substring_match = SubstringMatch()
        >>> lm_outputs = ["This is a cat .", "This is a dog ."]
        >>> references_list = [["cat", "dog"], ["mouse"]]
        >>> result = substring_match.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={'substring_match': 0.5},
            instance_details=[{'substring_match': True}, {'substring_match': False}]
        )
    """

    def __init__(self, mode: Literal["any", "all"] = "any") -> None:
        self.mode = mode
        if mode == "all":
            self.match_func = all
        elif mode == "any":
            self.match_func = any
        else:
            msg = f"mode must be 'any' or 'all', but got '{mode}'."
            raise ValueError(msg)

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if len(lm_outputs) != len(references_list):
            msg = (
                f"lm_outputs and references_list must have the same length, "
                f"but got {len(lm_outputs)} and {len(references_list)}."
            )
            raise ValueError(msg)

        match_list = [
            self.match_func(substring in lm_output for substring in expected_output)
            for lm_output, expected_output in zip(lm_outputs, references_list)
        ]

        score = 0.0
        if len(match_list):
            score = sum(match_list) / len(match_list)

        return MetricResult(
            {f"substring_match-{self.mode}": score},
            instance_details=[{"substring_match": match} for match in match_list],
        )
