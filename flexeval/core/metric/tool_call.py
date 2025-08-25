from __future__ import annotations

from collections import Counter

from flexeval.core.language_model.base import LMOutput

from .base import Metric, MetricResult
from .utils import validate_inputs


class ToolCallCount(Metric):
    """
    Metric to compute the ratio of different tool_call_validation_result values.
    """

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)

        # Calculate the finish_reason and validation statistics
        tool_call_validation_result_counter = Counter()
        for lm_output in lm_outputs:
            if not isinstance(lm_output, LMOutput):
                msg = "ToolCallCount expects lm_outputs to be an LMOutput, but received a different type."
                raise TypeError(msg)
            tool_call_validation_result_counter[lm_output.tool_call_validation_result] += 1

        total_count = sum(tool_call_validation_result_counter.values())
        summary = {}
        if total_count > 0:
            for validation_result, count in tool_call_validation_result_counter.items():
                summary[f"tool_call_validation_result_ratio-{validation_result}"] = count / total_count
        return MetricResult(
            summary=summary,
            instance_details=[
                {"tool_call_validation_result": lm_output.tool_call_validation_result} for lm_output in lm_outputs
            ],
        )
