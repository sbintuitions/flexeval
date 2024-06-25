from __future__ import annotations

import functools

from fuzzywuzzy import fuzz

from .base import Metric, MetricResult
from .string_processor import StringProcessor


class CharF1(Metric):
    """
    A metric that calculates how many characters in the output string are included
    in the characters of the expected output.
    If there are multiple expected outputs, the highest score is adopted.

    Args:
        processor: StringProcessor or list of Normalizers to apply to the model outputs before comparison.
            Unless reference_processor is specified, this processor will be applied to the references as well.
        reference_processor: StringProcessor or list of Normalizers to apply to the references before comparison.


    Examples:
        >>> from flexeval import CharF1
        >>> char_f1 = CharF1()
        >>> lm_outputs = ["abcd", "efgh"]
        >>> references_list = [["abcd", "ABCD"], ["efGH"]]
        >>> result = char_f1.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(summary={'char_f1': 0.75}, instance_details=[{'char_f1': 1.0}, {'char_f1': 0.5}])
    """

    def __init__(
        self,
        processor: StringProcessor | list[StringProcessor] | None = None,
        reference_processor: StringProcessor | list[StringProcessor] | None = None,
    ) -> None:
        if isinstance(processor, StringProcessor):
            processor = [processor]
        if isinstance(reference_processor, StringProcessor):
            reference_processor = [reference_processor]

        self.processors = processor
        self.reference_processors = reference_processor or processor

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if self.processors:
            lm_outputs = [functools.reduce(lambda x, norm: norm(x), self.processors, output) for output in lm_outputs]

        if self.reference_processors:
            references_list = [
                [functools.reduce(lambda x, norm: norm(x), self.reference_processors, ref) for ref in references]
                for references in references_list
            ]

        char_f1_scores: list[float] = []
        for lm_output, expected_output in zip(lm_outputs, references_list):
            score = max(fuzz.ratio(lm_output, o) for o in expected_output) / 100
            char_f1_scores.append(score)
        return MetricResult(
            {"char_f1": sum(char_f1_scores) / len(char_f1_scores)},
            instance_details=[{"char_f1": s} for s in char_f1_scores],
        )
