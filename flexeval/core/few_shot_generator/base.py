from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flexeval.core.chat_dataset import ChatDataset, ChatInstance
from flexeval.core.generation_dataset import GenerationDataset, GenerationInstance
from flexeval.core.multiple_choice_dataset import MultipleChoiceDataset, MultipleChoiceInstance

Dataset = GenerationDataset | MultipleChoiceDataset | ChatDataset
Instance = GenerationInstance | MultipleChoiceInstance | ChatInstance


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

    def with_seed_increment(self, seed_increment: int) -> FewShotGenerator:
        """
        Return a new `FewShotGenerator` instance to be used for a repeated evaluation run.

        The default implementation simply returns `self`, which is appropriate for
        generators that do not hold any sampling state (e.g., no internal RNG).
        Subclasses that hold a stateful RNG (e.g., `self._rnd = random.Random(seed)`)
        should override this method to return a new instance constructed with
        `seed + seed_increment`, so that each repeat's few-shot examples are
        reproducible from the seed alone, independent of how many times the
        original instance has already been sampled from.

        Args:
            seed_increment: The value to add to the original seed for the new instance.

        Returns:
            A `FewShotGenerator` instance to use for the repeated run.
        """
        return self

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
