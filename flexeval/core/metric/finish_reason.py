from __future__ import annotations

from collections import Counter

from flexeval.core.language_model.base import LMOutput

from .base import Metric, MetricResult
from .utils import validate_inputs


class FinishReasonCount(Metric):
    """
    Metric to compute the ratio of different finish_reason values.
    """

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)

        # Count finish_reason occurrences from messages
        finish_reason_counter = Counter()
        for lm_output in lm_outputs:
            if not isinstance(lm_output, LMOutput):
                msg = "FinishReasonMetric expects lm_outputs to be an LMOutput, but received a different type."
                raise TypeError(msg)
            finish_reason_counter[lm_output.finish_reason] += 1

        total_count = sum(finish_reason_counter.values())
        summary = {}
        if total_count > 0:
            for finish_reason, count in finish_reason_counter.items():
                summary[f"finish_reason_ratio-{finish_reason}"] = count / total_count

        return MetricResult(
            summary=summary, instance_details=[{"finish_reason": lm_output.finish_reason} for lm_output in lm_outputs]
        )
