from __future__ import annotations

import random
from typing import Any

from .base import Dataset, FewShotGenerator, Instance


class RandomFewShotGenerator(FewShotGenerator):
    def __init__(
        self,
        dataset: Dataset,
        num_shots: int,
        seed: int = 42,
        num_trials_to_avoid_leak: int = 3,
    ) -> None:
        super().__init__(num_trials_to_avoid_leak=num_trials_to_avoid_leak)

        if num_shots > len(dataset):
            msg = (
                f"`num_shots` should be less than or equal to the number of instances in `dataset`. "
                f"num_shots: {num_shots}, len(dataset): {len(dataset)}"
            )
            raise ValueError(msg)

        self._dataset = dataset
        self._num_shots = num_shots
        self._rnd = random.Random(seed)

    def _sample_instances(self, eval_inputs: list[dict[str, Any]] | dict[str, Any] | None = None) -> list[Instance]:
        sampled_indices = self._rnd.sample(range(len(self._dataset)), self._num_shots)
        return [self._dataset[i] for i in sampled_indices]
