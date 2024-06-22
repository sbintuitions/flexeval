from __future__ import annotations

from .base import Metric, MetricResult


class OutputLengthStats(Metric):
    """
    Compute statistics on the length of the outputs.

    Examples:
        >>> from flexeval import OutputLengthStats
        >>> output_length_stats = OutputLengthStats()
        >>> lm_outputs = ["123456", "123456789"]
        >>> result = output_length_stats.evaluate(lm_outputs)
        >>> print(result)
        MetricResult(
            summary={'avg_output_length': 7.5, 'max_output_length': 9, 'min_output_length': 6},
            instance_details=[{'output_length': 6}, {'output_length': 9}]
        )
    """

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]] | None = None,
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
