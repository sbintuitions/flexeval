from __future__ import annotations

from typing import Any

from .base import ResultRecorder


class WandBRecorder(ResultRecorder):
    """
    A class to record the results to Weights & Biases.

    Args:
        init_kwargs: The arguments for the `wandb.init` function.
            Please refer to [the official documentation](https://docs.wandb.ai/ref/python/init) for the details.
    """

    def __init__(
        self,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        init_kwargs = init_kwargs or {}
        self._wandb.init(**init_kwargs)

    def record_config(self, config: dict[str, Any], group: str | None = None) -> None:
        if group:
            self._wandb.config.update({group: config})
        else:
            self._wandb.config.update(config)

    def record_metrics(self, metrics: dict[str, Any], group: str | None = None) -> None:
        if group:
            self._wandb.summary.update({group: metrics})
        else:
            self._wandb.summary.update(metrics)

    def record_model_outputs(self, model_outputs: list[dict[str, Any]], group: str | None = None) -> None:
        table = self._wandb.Table(columns=list(model_outputs[0].keys()))

        for output in model_outputs:
            table.add_data(*output.values())

        table_name = "model_outputs" if group is None else f"{group}/model_outputs"
        self._wandb.log({table_name: table})

    def __del__(self) -> None:
        self._wandb.finish()
