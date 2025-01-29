from __future__ import annotations

import functools
from collections import Counter
from typing import Any

from .base import Metric, MetricResult
from .string_processor import StringProcessor


def get_most_repeated_pattern(text: str, threshold_length: int = 10) -> tuple[str, int]:
    special_chars = [" ", "=", "-", "/"]
    counter = Counter()
    for i in range(max(len(text) - threshold_length + 1, 1)):
        subtext = text[i : i + threshold_length]
        if any(subtext.startswith(c) or subtext.endswith(c) for c in special_chars):
            continue
        counter[subtext] += 1
    subtext, count = counter.most_common(1)[0]
    return subtext, count


class RepetitionCount(Metric):
    """
    A metric that counts the number of repetitions of the most repeated pattern in the model's output.

    Args:
        lm_output_processor: StringProcessor or list of Normalizers to apply to the model outputs before analysis.

    Examples:
        >>> from flexeval import RepetitionCount
        >>> repetition_count = RepetitionCount()
        >>> lm_outputs = ["hello hello hello hello hello hello hello hello hello hello"]
        >>> references_list = [[]]  # Not used for this metric
        >>> result = repetition_count.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={'avg_repetition_count': 2.5},
            instance_details=[
                {'most_repeated_pattern': 'abcabc', 'repetition_count': 3},
                {'most_repeated_pattern': 'defgdefg', 'repetition_count': 2}
            ]
        )
    """

    def __init__(
        self,
        count_threshold: int = 30,
        threshold_length: int = 10,
        lm_output_processor: StringProcessor | list[StringProcessor] | None = None,
    ) -> None:
        self.count_threshold = count_threshold
        self.threshold_length = threshold_length

        if isinstance(lm_output_processor, StringProcessor):
            lm_output_processor = [lm_output_processor]
        self.lm_output_processors = lm_output_processor

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],  # Not used in this metric
        task_inputs_list: list[dict[str, str]] | None = None,  # Not used in this metric
    ) -> MetricResult:
        if self.lm_output_processors:
            lm_outputs = [
                functools.reduce(lambda x, norm: norm(x), self.lm_output_processors, output) for output in lm_outputs
            ]

        repetition_details: list[dict[str, Any]] = []
        num_repetitions = 0
        for output in lm_outputs:
            most_repeated_pattern, count = get_most_repeated_pattern(output, threshold_length=self.threshold_length)
            is_repetition = count >= self.count_threshold
            repetition_details.append(
                {
                    "most_repeated_pattern": most_repeated_pattern,
                    "repetition_count": count,
                    "is_repetition": is_repetition,
                }
            )
            num_repetitions += int(is_repetition)

        repetition_rate = num_repetitions / len(lm_outputs)

        return MetricResult(
            summary={"repetition_ratio": repetition_rate},
            instance_details=repetition_details,
        )
