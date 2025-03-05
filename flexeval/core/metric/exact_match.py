from __future__ import annotations

import functools

from flexeval.core.string_processor import StringProcessor

from .base import Metric, MetricResult
from .utils import aggregate_category_wise_scores


class ExactMatch(Metric):
    """
    Exact match metric.
    If there are multiple references, the output is considered correct if it matches any of the references.

    Args:
        lm_output_processor:
            StringProcessor or a list of StringProcessor to be applied to the model outputs before comparison.
        reference_processor: StringProcessor or list of StringProcessor to apply to the references before comparison.

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
        lm_output_processor: StringProcessor | list[StringProcessor] | None = None,
        reference_processor: StringProcessor | list[StringProcessor] | None = None,
        category_key: str | None = None,
    ) -> None:
        if isinstance(lm_output_processor, StringProcessor):
            lm_output_processor = [lm_output_processor]
        if isinstance(reference_processor, StringProcessor):
            reference_processor = [reference_processor]

        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor
        self.category_key = category_key

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

        if self.lm_output_processors:
            lm_outputs = [
                functools.reduce(lambda x, norm: norm(x), self.lm_output_processors, output) for output in lm_outputs
            ]

        if self.reference_processors:
            references_list = [
                [functools.reduce(lambda x, norm: norm(x), self.reference_processors, ref) for ref in references]
                for references in references_list
            ]

        exact_match_list = [
            lm_output in expected_output for lm_output, expected_output in zip(lm_outputs, references_list)
        ]
        summary = {"exact_match": sum(exact_match_list) / len(exact_match_list)}

        if self.category_key:
            categories = [task_input[self.category_key] for task_input in task_inputs_list]
            category_wise_scores = aggregate_category_wise_scores(exact_match_list, categories)
            for category, category_wise_score in category_wise_scores.items():
                summary[f"exact_match/{category}"] = category_wise_score

        return MetricResult(
            summary,
            instance_details=[{"exact_match": s} for s in exact_match_list],
        )
