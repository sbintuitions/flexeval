from __future__ import annotations

from flexeval.core.string_processor import StringProcessor

from .base import Metric, MetricResult
from .utils import aggregate_category_wise_scores, apply_string_processors, validate_inputs


class ExactMatch(Metric):
    """
    Exact match metric.
    If there are multiple references, the output is considered correct if it matches any of the references.

    Args:
        lm_output_processor:
            StringProcessor or a list of StringProcessor to be applied to the model outputs before comparison.
        reference_processor: StringProcessor or list of StringProcessor to apply to the references before comparison.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in extra_info.
        metric_key: The metric name to store the mean score in metrics.json.
            Use this if you try multiple ExactMatch in differenct settings at once, e.g. difference string processors.

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
        metric_key: str = "exact_match",
    ) -> None:
        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor
        self.category_key = category_key
        self.metric_key = metric_key

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)

        # Normalize text data
        lm_outputs = [apply_string_processors(output, self.lm_output_processors) for output in lm_outputs]
        references_list = [
            [apply_string_processors(ref, self.reference_processors) for ref in references]
            for references in references_list
        ]

        # Compute metrics
        exact_match_list = [
            lm_output in expected_output for lm_output, expected_output in zip(lm_outputs, references_list)
        ]
        summary = {self.metric_key: sum(exact_match_list) / len(exact_match_list)}

        if self.category_key:
            categories = [extra_info[self.category_key] for extra_info in extra_info_list]
            category_wise_scores = aggregate_category_wise_scores(exact_match_list, categories)
            for category, category_wise_score in category_wise_scores.items():
                summary[f"{self.metric_key}/{category}"] = category_wise_score

        return MetricResult(
            summary,
            instance_details=[{self.metric_key: s} for s in exact_match_list],
        )
