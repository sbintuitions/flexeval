from __future__ import annotations

from .base import Metric, MetricResult


class OutputLengthStats(Metric):
    """Compute statistics on the length of the outputs."""

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        output_length_list = [len(output) for output in lm_outputs]
        return MetricResult(
            {
                "avg_output_length": sum(output_length_list) / len(output_length_list),
                "max_output_length": max(output_length_list),
                "min_output_length": min(output_length_list),
            },
            instance_details=[{"output_length": s} for s in output_length_list],
        )
