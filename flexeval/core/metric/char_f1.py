from __future__ import annotations

from fuzzywuzzy import fuzz

from flexeval.core.metric.utils import aggregate_category_wise_scores, apply_string_processors, validate_inputs
from flexeval.core.string_processor import StringProcessor

from .base import Metric, MetricResult


class CharF1(Metric):
    """
    A metric that calculates how many characters in the output string are included
    in the characters of the expected output.
    If there are multiple expected outputs, the highest score is adopted.

    Args:
        lm_output_processor: StringProcessor or list of Normalizers to apply to the model outputs before comparison.
        reference_processor: StringProcessor or list of Normalizers to apply to the references before comparison.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in extra_info.

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
        lm_output_processor: StringProcessor | list[StringProcessor] | None = None,
        reference_processor: StringProcessor | list[StringProcessor] | None = None,
        category_key: str | None = None,
    ) -> None:
        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor
        self.category_key = category_key

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
        char_f1_scores: list[float] = []
        for lm_output, expected_output in zip(lm_outputs, references_list):
            score = max(fuzz.ratio(lm_output, o) for o in expected_output) / 100
            char_f1_scores.append(score)

        summary = {"char_f1": sum(char_f1_scores) / len(char_f1_scores)}

        if self.category_key:
            categories = [extra_info[self.category_key] for extra_info in extra_info_list]
            category_wise_scores = aggregate_category_wise_scores(char_f1_scores, categories)
            for category, category_wise_score in category_wise_scores.items():
                summary[f"char_f1/{category}"] = category_wise_score

        return MetricResult(
            summary,
            instance_details=[{"char_f1": s} for s in char_f1_scores],
        )
