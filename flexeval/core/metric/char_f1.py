from __future__ import annotations

import functools

from fuzzywuzzy import fuzz

from .base import Metric, MetricResult
from .normalizer import Normalizer


class CharF1(Metric):
    """
    A metric that calculates how many characters in the output string are included
    in the characters of the expected output.
    If there are multiple expected outputs, the highest score is adopted.

    Args:
        normalizer: Normalizer or list of Normalizers to apply to the model outputs before comparison.
            Unless reference_normalizer is specified, this normalizer will be applied to the references as well.
        reference_normalizer: Normalizer or list of Normalizers to apply to the references before comparison.


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
        normalizer: Normalizer | list[Normalizer] | None = None,
        reference_normalizer: Normalizer | list[Normalizer] | None = None,
    ) -> None:
        if isinstance(normalizer, Normalizer):
            normalizer = [normalizer]
        if isinstance(reference_normalizer, Normalizer):
            reference_normalizer = [reference_normalizer]

        self.normalizers = normalizer
        self.reference_normalizers = reference_normalizer or normalizer

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if self.normalizers:
            lm_outputs = [functools.reduce(lambda x, norm: norm(x), self.normalizers, output) for output in lm_outputs]

        if self.reference_normalizers:
            references_list = [
                [functools.reduce(lambda x, norm: norm(x), self.reference_normalizers, ref) for ref in references]
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
