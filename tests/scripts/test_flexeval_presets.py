from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_flexeval_presets_cli() -> None:
    os.environ["PRESET_CONFIG_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")
    os.environ["PRESET_CONFIG_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")
    os.environ["PRESET_CONFIG_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

    for valid_command in [
        ["flexeval_presets"],
        ["flexeval_presets", "generation"],
    ]:
        result = subprocess.run(valid_command, check=False)
        assert result.returncode == os.EX_OK

    for invalid_command in [
        ["flexeval_presets", "no_such_preset"],
        ["flexeval_presets", "--no_such_option"],
    ]:
        result = subprocess.run(invalid_command, check=False)
        assert result.returncode != os.EX_OK
