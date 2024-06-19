from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, TypeVar

from jsonargparse import ArgumentParser
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


class ConfigNameResolver:
    """
    This class resolves the path to the jsonnet config file from the name of the preset config.
    """

    def __init__(self, config_directory: str | PathLike[str], config_glob: str = "*.jsonnet") -> None:
        self._name_to_path: dict[str, Path] = {}
        for config_path in Path(config_directory).rglob(config_glob):
            self._name_to_path[config_path.stem] = config_path

    def __call__(self, config_name_or_path: str) -> str | None:
        # When the argument parser gets both the path and the name of the preset config,
        # the path is given as a string and not loaded as parameters.
        # We catch this case here and return the path as it is.
        if Path(config_name_or_path).exists():
            return config_name_or_path

        if config_name_or_path not in self._name_to_path:
            return None
        return str(self._name_to_path[config_name_or_path])


Module = TypeVar("Module", bound=Any)


def instantiate_module_from_path(
    config_path: str,
    module_type: type[Module],
    overrides: dict[str, Any] | None = None,
) -> Module:
    parser = ArgumentParser(parser_mode="jsonnet")
    parser.add_argument("--module", type=module_type, required=True, enable_path=True)

    args_to_parse = ["--module", config_path]
    overrides = overrides or {}
    for key, value in overrides.items():
        args_to_parse += [f"--module.{key}", value]

    args = parser.parse_args(args_to_parse)
    instantiated_config = parser.instantiate_classes(args)
    return instantiated_config.module


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
