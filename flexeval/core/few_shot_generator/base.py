from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

from flexeval.core.chat_dataset import ChatDataset, ChatInstance
from flexeval.core.generation_dataset import GenerationDataset, GenerationInstance
from flexeval.core.multiple_choice_dataset import MultipleChoiceDataset, MultipleChoiceInstance

Dataset = Union[GenerationDataset, MultipleChoiceDataset, ChatDataset]
Instance = Union[GenerationInstance, MultipleChoiceInstance, ChatInstance]


class FewShotGenerator(ABC):
    def __init__(self, num_trials_to_avoid_leak: int) -> None:
        self._num_trials_to_avoid_leak = num_trials_to_avoid_leak

    @abstractmethod
    def _sample_instances(self, eval_inputs: list[dict[str, Any]] | dict[str, Any] | None = None) -> list[Instance]:
        """
        Sample instances for few-shot learning.
        This method should be implemented in the derived class.
        """
        raise NotImplementedError

    def __call__(self, eval_inputs: list[dict[str, Any]] | dict[str, Any] | None = None) -> list[Instance]:
        """
        Sample instances for few-shot learning.
        This method calls `_sample_instances` and
        checks if the sampled instances have the same inputs as the evaluation instance.

        Args:
            eval_inputs: The inputs of the evaluation instance.
                This is used to avoid data leakage
                by checking if the sampled instances have the same inputs as the evaluation instance.

        Returns:
            A list of instances for few-shot learning.
        """
        sampled_instances = self._sample_instances(eval_inputs=eval_inputs)

        # check if the sampled instances are the same as the eval_instance
        if self._num_trials_to_avoid_leak and eval_inputs is not None:
            for _ in range(self._num_trials_to_avoid_leak):
                if all(sampled.inputs != eval_inputs for sampled in sampled_instances):
                    return sampled_instances
                # retry sampling
                sampled_instances = self._sample_instances(eval_inputs=eval_inputs)

            msg = (
                f"Few-shot instance has the same inputs as the evaluation instance, "
                f"which indicates a data leak. "
                f"Failed to sample a different instance after {self._num_trials_to_avoid_leak} trials."
            )
            raise ValueError(msg)

        return sampled_instances
