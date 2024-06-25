from __future__ import annotations

from typing import Any, Sequence

from .base import ResultRecorder


class WandBRecorder(ResultRecorder):
    """
    A class to record the results to Weights & Biases.

    The arguments are copied from the `wandb.init` function with the version 0.17.2.
    https://docs.wandb.ai/ref/python/init
    """

    def __init__(
        self,
        job_type: str | None = None,
        dir: str | None = None,  # noqa: A002
        config: dict | str | None = None,
        project: str | None = None,
        entity: str | None = None,
        reinit: bool | None = None,
        tags: Sequence | None = None,
        group: str | None = None,
        name: str | None = None,
        notes: str | None = None,
        magic: dict | str | bool | None = None,
        config_exclude_keys: list[str] | None = None,
        config_include_keys: list[str] | None = None,
        anonymous: str | None = None,
        mode: str | None = None,
        allow_val_change: bool | None = None,
        resume: bool | str | None = None,
        force: bool | None = None,
        tensorboard: bool | None = None,
        sync_tensorboard: bool | None = None,
        monitor_gym: bool | None = None,
        save_code: bool | None = None,
        id: str | None = None,  # noqa: A002
        fork_from: str | None = None,
        resume_from: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb

        self._wandb.init(
            job_type=job_type,
            dir=dir,
            config=config,
            project=project,
            entity=entity,
            reinit=reinit,
            tags=tags,
            group=group,
            name=name,
            notes=notes,
            magic=magic,
            config_exclude_keys=config_exclude_keys,
            config_include_keys=config_include_keys,
            anonymous=anonymous,
            mode=mode,
            allow_val_change=allow_val_change,
            resume=resume,
            force=force,
            tensorboard=tensorboard,
            sync_tensorboard=sync_tensorboard,
            monitor_gym=monitor_gym,
            save_code=save_code,
            id=id,
            fork_from=fork_from,
            resume_from=resume_from,
            settings=settings,
        )

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
