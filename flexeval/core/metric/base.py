from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from flexeval.core.language_model.base import LMOutput


@dataclass
class MetricResult:
    """
    A dataclass representing the result of a metric evaluation.
    """

    summary: dict[str, Any]
    """
    Summary containing aggregated metric values.
    """
    instance_details: list[dict[str, Any]] | None = None
    """
    A list of evaluate details for each instance.
    Useful for error analysis.
    """


class Metric(ABC):
    """
    Base class for metrics.

    Subclasses must implement the `evaluate` method to perform metric computation.
    Use utility functions from `flexeval.core.metric.utils` for common patterns
    like string processing and category-wise aggregation.
    """

    @abstractmethod
    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        """
        Evaluate the outputs of `LanguageModel` against the references.

        Args:
            lm_outputs: List of model outputs (strings or LMOutput objects).
            references_list: List of reference outputs.
            extra_info_list: List of task inputs and some extra information.
        """

    def cleanup_resources(self) -> None:  # noqa: B027
        """
        Clean up resources if necessary.
        This method is called when the metric is no longer needed.
        """
        pass  # noqa: PIE790
