from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .base import FewShotGenerator, GenerationDataset, GenerationInstance


class BalancedFewShotGenerator(FewShotGenerator):
    def __init__(
        self,
        dataset: GenerationDataset,
        num_shots: int,
        seed: int = 42,
        num_trials_to_avoid_leak: int = 3,
    ) -> None:
        super().__init__(num_trials_to_avoid_leak=num_trials_to_avoid_leak)
        if not isinstance(dataset, GenerationDataset):
            msg = "BalancedFewShotGenerator only supports GenerationDataset"
            raise TypeError(msg)

        if num_shots > len(dataset):
            msg = (
                f"`num_shots` should be less than or equal to the number of instances in `dataset`. "
                f"num_shots: {num_shots}, len(dataset): {len(dataset)}"
            )
            raise ValueError(msg)

        self.dataset = dataset
        self.num_shots = num_shots
        self._rnd = random.Random(seed)

        # Separate instances by label
        # Here we assume that the label is the first element of references of the instance.
        label_to_ids: dict[str, list[int]] = defaultdict(list)
        for i, instance in enumerate(dataset):
            label_to_ids[instance.references[0]].append(i)
        self._label_to_ids = label_to_ids

    def _sample_instances(
        self,
        eval_inputs: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> list[GenerationInstance]:
        # Shuffle labels
        labels = list(self._label_to_ids.keys())
        self._rnd.shuffle(labels)

        # Evenly distribute num_samples to each label
        num_samples_list = [self.num_shots // len(labels)] * len(labels)
        remaining_samples = self.num_shots % len(labels)
        for i in range(remaining_samples):
            num_samples_list[i] += 1

        # Sample instances from each label
        sampled_indices: list[int] = []
        for label, num_samples_for_the_label in zip(labels, num_samples_list):
            sampled_indices += self._rnd.sample(
                self._label_to_ids[label],
                num_samples_for_the_label,
            )
        self._rnd.shuffle(sampled_indices)

        return [self.dataset[i] for i in sampled_indices]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset}, num_shots={self.num_shots})"
