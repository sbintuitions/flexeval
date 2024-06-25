from __future__ import annotations

import functools

from .base import Metric, MetricResult
from .string_processor import StringProcessor


class ExactMatch(Metric):
    """
    Exact match metric.
    If there are multiple references, the output is considered correct if it matches any of the references.

    Args:
        processor: StringProcessor or a list of StringProcessor to be applied to the model outputs before comparison.
            Unless reference_processor is specified, this processor will be applied to the references as well.
        reference_processor: StringProcessor or list of Normalizers to apply to the references before comparison.

    Examples:
        >>> from flexeval import ExactMatch
        >>> exact_match = ExactMatch()
        >>> lm_outputs = ["ABC", "DEF"]
        >>> references_list = [["ABC"], ["DEFG"]]
        >>> result = exact_match.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={"exact_match": 0.5},
            instance_details=[{"exact_match": True}, {"exact_match": False}],
        )
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
        if len(lm_outputs) != len(references_list):
            msg = (
                f"Number of model outputs ({len(lm_outputs)}) and number of references ({len(references_list)}) "
                "should be the same."
            )
            raise ValueError(msg)

        if self.processors:
            lm_outputs = [functools.reduce(lambda x, norm: norm(x), self.processors, output) for output in lm_outputs]

        if self.reference_processors:
            references_list = [
                [functools.reduce(lambda x, norm: norm(x), self.reference_processors, ref) for ref in references]
                for references in references_list
            ]

        exact_match_list = [
            lm_output in expected_output for lm_output, expected_output in zip(lm_outputs, references_list)
        ]

        return MetricResult(
            {"exact_match": sum(exact_match_list) / len(exact_match_list)},
            instance_details=[{"exact_match": s} for s in exact_match_list],
        )
