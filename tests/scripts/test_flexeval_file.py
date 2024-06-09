from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from flexeval.scripts.common import OUTPUTS_FILE_NAME

from .test_flexeval_lm import CHAT_RESPONSE_CMD, GENERATION_CMD, check_if_eval_results_are_correctly_saved


@pytest.mark.parametrize(
    ("flexeval_lm_command", "metrics_args"),
    [
        (CHAT_RESPONSE_CMD, ["--metrics", "ExactMatch"]),
        (GENERATION_CMD, ["--metrics", "ExactMatch", "--metrics+=CharF1"]),
        (CHAT_RESPONSE_CMD, ["--metrics", "exact_match"]),
        (GENERATION_CMD, ["--metrics", "exact_match", "--metrics+=CharF1"]),
    ],
)
def test_if_outputs_from_flexval_lm_can_be_passed_to_flexeval_file(
    flexeval_lm_command: list[str],
    metrics_args: list[str],
) -> None:
    os.environ["PRESET_CONFIG_METRIC_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

    with tempfile.TemporaryDirectory() as f:
        result_from_flexeval_lm = subprocess.run([*flexeval_lm_command, "--save_dir", f], check=False)
        assert result_from_flexeval_lm.returncode == os.EX_OK

        output_file_path = Path(f) / OUTPUTS_FILE_NAME

        save_path_flexeval_file = Path(f) / "flexeval_file"
        # fmt: off
        result_from_flexeval_file = subprocess.run(
            [  # noqa: S607
                "flexeval_file",
                "--eval_file", str(output_file_path),
                *metrics_args,
                "--save_dir", str(save_path_flexeval_file),
            ],
            check=False,
        )
        # fmt: on
        assert result_from_flexeval_file.returncode == os.EX_OK

        check_if_eval_results_are_correctly_saved(save_path_flexeval_file)
