from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any

from loguru import logger

from flexeval.core.utils.json_util import Base64TruncatingJSONEncoder

from .base import ResultRecorder

METRIC_FILE_NAME = "metrics.json"
OUTPUTS_FILE_NAME = "outputs.jsonl"
CONFIG_FILE_NAME = "config.json"
LOCK_FILE_NAME = ".lock"
LOCK_STEAL_GUARD_FILE_NAME = ".lock.steal"


def save_jsonl(
    data: Iterable[dict[str, Any] | list[Any]],
    save_path: str | PathLike[str],
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        for d in data:
            dump_line = json.dumps(d, ensure_ascii=False, cls=Base64TruncatingJSONEncoder)
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
        force: Overwrite the existing results.
        lock_timeout: A lock file (`.lock`) that has not been updated for this many seconds
            is regarded as stale (left by a dead process) and can be taken over.
        heartbeat_interval: The interval in seconds to update the lock file while it is held.
            Defaults to `lock_timeout / 20`.
    """

    def __init__(
        self,
        output_dir: str,
        force: bool = False,
        lock_timeout: float = 300.0,
        heartbeat_interval: float | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.force = force
        self.lock_timeout = lock_timeout
        self.heartbeat_interval = heartbeat_interval if heartbeat_interval is not None else lock_timeout / 20
        self._heartbeats: dict[Path, tuple[threading.Thread, threading.Event]] = {}

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

    def is_metrics_saved(self, group: str | None = None) -> bool:
        output_dir = self.output_dir
        if group is not None:
            output_dir = self.output_dir / group
        return (output_dir / METRIC_FILE_NAME).exists()

    def _lock_path(self, group: str | None) -> Path:
        output_dir = self.output_dir
        if group is not None:
            output_dir = self.output_dir / group
        return output_dir / LOCK_FILE_NAME

    @staticmethod
    def _create_lock_file(lock_path: Path) -> bool:
        try:
            # O_EXCL makes the creation atomic, so only one process can succeed.
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.close(fd)
        return True

    def _is_stale(self, path: Path) -> bool:
        try:
            return time.time() - path.stat().st_mtime > self.lock_timeout
        except FileNotFoundError:
            return True

    def _start_heartbeat(self, lock_path: Path) -> None:
        stop_event = threading.Event()

        def _update_mtime() -> None:
            while not stop_event.wait(self.heartbeat_interval):
                try:
                    os.utime(lock_path)
                except OSError:  # noqa: PERF203
                    logger.warning(f"Failed to update the lock file: {lock_path}")

        thread = threading.Thread(target=_update_mtime, daemon=True)
        thread.start()
        self._heartbeats[lock_path] = (thread, stop_event)

    def _take_over_stale_lock(self, lock_path: Path) -> bool:
        # Only one process can hold the guard, so that a process does not remove the lock
        # that another process has just taken over.
        guard_path = lock_path.with_name(LOCK_STEAL_GUARD_FILE_NAME)
        if not self._create_lock_file(guard_path):
            if self._is_stale(guard_path):
                guard_path.unlink(missing_ok=True)
            return False
        try:
            if not self._is_stale(lock_path):
                return False
            logger.warning(f"Taking over the stale lock: {lock_path}")
            lock_path.unlink(missing_ok=True)
            return self._create_lock_file(lock_path)
        finally:
            guard_path.unlink(missing_ok=True)

    def acquire_lock(self, group: str | None = None) -> bool:
        lock_path = self._lock_path(group)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._create_lock_file(lock_path) and not (
            self._is_stale(lock_path) and self._take_over_stale_lock(lock_path)
        ):
            return False
        self._start_heartbeat(lock_path)
        return True

    def release_lock(self, group: str | None = None) -> None:
        lock_path = self._lock_path(group)
        heartbeat = self._heartbeats.pop(lock_path, None)
        if heartbeat is not None:
            thread, stop_event = heartbeat
            stop_event.set()
            thread.join()
        lock_path.unlink(missing_ok=True)
