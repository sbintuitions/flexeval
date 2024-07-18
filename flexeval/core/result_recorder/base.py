from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ResultRecorder(ABC):
    """
    An abstract base class for recording experiment results, including configuration,
    metrics, and model outputs.

    This class defines the interface for different result recording implementations,
    such as saving to a local directory, uploading to wandb, or integrating with MLflow.
    """

    @abstractmethod
    def record_config(self, config: dict[str, Any], group: str | None = None) -> None:
        """
        Record the configuration parameters of the experiment.

        Args:
            config: A dictionary containing the configuration
                parameters of the evaluation.
            group: An optional group name to organize the configuration.
        """

    @abstractmethod
    def record_metrics(self, metrics: dict[str, Any], group: str | None = None) -> None:
        """
        Record the evaluation metrics of the experiment.

        Args:
            metrics: A dictionary containing the evaluation metrics,
                where keys are metric names and values are the corresponding results.
            group: An optional group name to organize the metrics.
        """

    @abstractmethod
    def record_model_outputs(self, model_outputs: list[dict[str, Any]], group: str | None = None) -> None:
        """
        Record the outputs generated by the model during evaluation.

        Args:
            model_outputs: A list of dictionaries, where each
                dictionary represents a single model output. The structure of these
                dictionaries may vary depending on the specific model and task.
            group: An optional group name to organize the model outputs.
        """