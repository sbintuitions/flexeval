from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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
    """

    @abstractmethod
    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        """
        Evaluate the outputs of `LanguageModel` against the references.

        Args:
            lm_outputs: List of model outputs.
            references_list: List of reference outputs.
            task_inputs_list: List of task inputs.
        """
        raise NotImplementedError
