from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
from typing_extensions import Self

METRIC_FILE_NAME = "metrics.json"
OUTPUTS_FILE_NAME = "outputs.jsonl"
CONFIG_FILE_NAME = "config.json"


def raise_error_if_results_already_exist(save_dir: str | PathLike[str]) -> None:
    for file_name in [METRIC_FILE_NAME, OUTPUTS_FILE_NAME, CONFIG_FILE_NAME]:
        if (Path(save_dir) / file_name).exists():
            msg = (
                f"`{Path(save_dir) / file_name}` already exists. If you want to overwrite it, "
                f"please specify `--force true`."
            )
            raise FileExistsError(msg)


def save_json(json_dict: dict[str, Any], save_path: str | PathLike[str]) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False, default=str)


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


def load_jsonl(path: str | PathLike[str]) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_env_metadata() -> dict[str, Any]:
    git_hash: str | None = None
    with contextlib.suppress(subprocess.CalledProcessError):
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()  # noqa: S607

    return {
        "python_version": sys.version,
        "flexeval_version": importlib_metadata.version("flexeval"),
        "git_hash": git_hash,
        "installed_packages": sorted(
            [f"{dist.metadata['Name']}=={dist.version}" for dist in importlib_metadata.distributions()],
        ),
    }


class Timer:
    def __enter__(self) -> Self:
        self.start = time.perf_counter()
        return self

    def __exit__(self, type_, value, traceback) -> None:  # noqa: ANN001
        self.time = time.perf_counter() - self.start


def override_jsonargparse_params(params: dict[str, Any], nested_key: str, new_value: Any) -> dict[str, Any]:  # noqa: ANN401
    keys = nested_key.split(".")
    current_params = params
    for k in keys[:-1]:  # Navigate to the last dictionary
        if "init_args" in current_params:
            current_params = current_params["init_args"]
        current_params = current_params[k]

    if "init_args" in current_params:
        current_params = current_params["init_args"]
    current_params[keys[-1]] = new_value
    return params
