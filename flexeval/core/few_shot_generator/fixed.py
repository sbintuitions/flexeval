from __future__ import annotations

from typing import Any

from .base import ChatInstance, FewShotGenerator, GenerationInstance, Instance, MultipleChoiceInstance


class FixedFewShotGenerator(FewShotGenerator):
    def __init__(self, instance_class: str, instance_params: list[dict[str, Any]]) -> None:
        super().__init__(num_trials_to_avoid_leak=0)

        if instance_class == "GenerationInstance":
            instance_init = GenerationInstance
        elif instance_class == "MultipleChoiceInstance":
            instance_init = MultipleChoiceInstance
        elif instance_class == "ChatInstance":
            instance_init = ChatInstance
        else:
            msg = f"Unknown instance class: {instance_class}"
            raise ValueError(msg)

        self.instances = [instance_init(**params) for params in instance_params]

    def _sample_instances(self, eval_inputs: list[dict[str, Any]] | dict[str, Any] | None = None) -> list[Instance]:
        return self.instances

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(instances={self.instances})"
