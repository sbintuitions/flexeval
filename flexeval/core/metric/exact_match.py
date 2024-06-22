from __future__ import annotations

from .base import Metric, MetricResult
from .normalizer import Normalizer


class ExactMatch(Metric):
    """
    Exact match metric.
    If there are multiple references, the output is considered correct if it matches any of the references.

    Args:
        normalizer: An instance of `Normalizer` to normalize the input and output strings.

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

    def __init__(self, normalizer: Normalizer | None = None) -> None:
        self.normalizer = normalizer

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

        if self.normalizer:
            lm_outputs = [self.normalizer.normalize(output) for output in lm_outputs]
            references_list = [
                [self.normalizer.normalize(reference) for reference in references] for references in references_list
            ]

        exact_match_list = [
            lm_output in expected_output for lm_output, expected_output in zip(lm_outputs, references_list)
        ]

        return MetricResult(
            {"exact_match": sum(exact_match_list) / len(exact_match_list)},
            instance_details=[{"exact_match": s} for s in exact_match_list],
        )
