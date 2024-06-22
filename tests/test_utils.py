from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from flexeval import EvalSetup, Metric

TEST_CONFIG_DIR = Path(__file__).parent / "dummy_modules" / "configs"


@pytest.mark.parametrize(
    ("module_type", "config_path", "overrides"),
    [
        (EvalSetup, str(TEST_CONFIG_DIR / "generation.jsonnet"), None),
        (EvalSetup, str(TEST_CONFIG_DIR / "generation.jsonnet"), {"batch_size": 100}),
        (EvalSetup, str(TEST_CONFIG_DIR / "multiple_choice.jsonnet"), None),
        (EvalSetup, str(TEST_CONFIG_DIR / "perplexity.jsonnet"), None),
        (Metric, str(TEST_CONFIG_DIR / "exact_match.jsonnet"), None),
        (EvalSetup, "generation", None),
        (EvalSetup, "multiple_choice", None),
        (EvalSetup, "perplexity", None),
        (Metric, "exact_match", None),
    ],
)
def test_instantiate_from_config(module_type: type, config_path: str, overrides: dict[str, Any] | None) -> None:
    os.environ["PRESET_CONFIG_DIR"] = str(TEST_CONFIG_DIR)
    from flexeval.utils import instantiate_from_config

    module = instantiate_from_config(config_path=config_path, overrides=overrides)
    assert isinstance(module, module_type)

    if overrides:
        for key, value in overrides.items():
            assert getattr(module, key) == value
