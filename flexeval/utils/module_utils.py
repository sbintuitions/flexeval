from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

import _jsonnet
from jsonargparse import ArgumentParser

import flexeval

Module = TypeVar("Module", bound=Any)


def instantiate_from_config(
    config_path: str,
    overrides: dict[str, Any] | None = None,
) -> Module:
    """
    Instantiates a module from a jsonnet config file.

    Args:
        config_path: The path to the jsonnet config file.
        overrides: A dictionary of overrides to apply to the config.

    Returns:
        The instantiated module.

    Examples:
        >>> from flexeval import instantiate_from_config
        >>> eval_setup = instantiate_from_config("aio")
    """
    resolved_config_path = ConfigNameResolver()(config_path)

    if resolved_config_path is None:
        msg = f'Config name "{config_path}" not found in the specified path nor in the preset config directories.'
        raise ValueError(msg)

    config = json.loads(_jsonnet.evaluate_file(resolved_config_path))
    module_class = getattr(flexeval, config["class_path"])

    parser = ArgumentParser(parser_mode="jsonnet")
    parser.add_argument("--module", type=module_class, required=True, enable_path=True)

    args_to_parse = ["--module", resolved_config_path]
    overrides = overrides or {}
    for key, value in overrides.items():
        args_to_parse += [f"--module.{key}", str(value)]

    args = parser.parse_args(args_to_parse)
    instantiated_config = parser.instantiate_classes(args)
    return instantiated_config.module


class ConfigNameResolver:
    """
    This class resolves the path to the jsonnet config file from the name of the preset config.
    """

    def __init__(self, config_glob: str = "*.jsonnet") -> None:
        config_directory = os.environ.get(
            "PRESET_CONFIG_DIR",
            Path(__file__).parent.parent / "preset_configs",
        )

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
