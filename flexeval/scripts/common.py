from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from os import PathLike
from typing import Any

from typing_extensions import Self


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
