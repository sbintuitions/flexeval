from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from .base import ResultRecorder

METRIC_FILE_NAME = "metrics.json"
OUTPUTS_FILE_NAME = "outputs.jsonl"
CONFIG_FILE_NAME = "config.json"


def save_jsonl(
    data: Iterable[dict[str, Any] | list[Any]],
    save_path: str | PathLike[str],
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        for d in data:
            dump_line = json.dumps(d, ensure_ascii=False, default=str)
            try:
                f.write(f"{dump_line}\n")
            except UnicodeEncodeError:
                logger.warning("Failed to write the following line")
                logger.warning(dump_line)


def save_json(json_dict: dict[str, Any], save_path: str | PathLike[str]) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False, default=str)


class LocalRecorder(ResultRecorder):
    """
    A class to record the results in JSON format.

    Args:
        output_dir: The directory to save the results.
    """

    def __init__(self, output_dir: str, force: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.force = force

    @staticmethod
    def _check_output_dir_exists(output_dir: str | PathLike[str], checked_files: list[str]) -> None:
        output_dir = Path(output_dir)
        for file_name in checked_files:
            if (output_dir / file_name).exists():
                msg = (
                    f"`{output_dir / file_name}` already exists. If you want to overwrite it, "
                    f"please specify `--force true` from CLI or `force=True` when initializing the recorder."
                )
                raise FileExistsError(msg)

    def record_config(self, config: dict[str, Any], group: str | None = None) -> None:
        output_dir = self.output_dir
        if group is not None:
            output_dir = self.output_dir / group

        if not self.force:
            self._check_output_dir_exists(output_dir, [CONFIG_FILE_NAME])

        save_json(config, output_dir / CONFIG_FILE_NAME)
        logger.info(f"Saved the config to {output_dir / CONFIG_FILE_NAME}")

    def record_metrics(self, metrics: dict[str, Any], group: str | None = None) -> None:
        output_dir = self.output_dir
        if group is not None:
            output_dir = self.output_dir / group

        if not self.force:
            self._check_output_dir_exists(output_dir, [METRIC_FILE_NAME])

        save_json(metrics, output_dir / METRIC_FILE_NAME)
        logger.info(f"Saved the metrics to {output_dir / METRIC_FILE_NAME}")

    def record_model_outputs(self, model_outputs: list[dict[str, Any]], group: str | None = None) -> None:
        output_dir = self.output_dir
        if group is not None:
            output_dir = output_dir / group

        if not self.force:
            self._check_output_dir_exists(output_dir, [OUTPUTS_FILE_NAME])

        save_jsonl(model_outputs, output_dir / OUTPUTS_FILE_NAME)
        logger.info(f"Saved the outputs to {output_dir / OUTPUTS_FILE_NAME}")
