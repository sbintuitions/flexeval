from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from flexeval.core.result_recorder.local_recorder import OUTPUTS_FILE_NAME

from .test_flexeval_lm import CHAT_RESPONSE_CMD, GENERATION_CMD, check_if_eval_results_are_correctly_saved


@pytest.mark.parametrize(
    "flexeval_lm_command",
    [CHAT_RESPONSE_CMD, GENERATION_CMD],
)
@pytest.mark.parametrize(
    "metrics_args",
    [
        ["--metrics", "ExactMatch"],
        ["--metrics", "exact_match"],
        ["--metrics", "ExactMatch", "--metrics+=CharF1"],
    ],
)
@pytest.mark.parametrize(
    "override_args",
    [[], ["--metrics.lm_output_processor", "AIONormalizer"]],
)
def test_if_outputs_from_flexval_lm_can_be_passed_to_flexeval_file(
    flexeval_lm_command: list[str],
    metrics_args: list[str],
    override_args: list[str],
) -> None:
    os.environ["PRESET_CONFIG_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

    with tempfile.TemporaryDirectory() as f:
        # prepare the output file from flexeval_lm
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
                *override_args,
                "--save_dir", str(save_path_flexeval_file),
            ],
            check=False,
        )
        # fmt: on
        assert result_from_flexeval_file.returncode == os.EX_OK

        check_if_eval_results_are_correctly_saved(save_path_flexeval_file)


def test_with_eval_data_loader() -> None:
    os.environ["PRESET_CONFIG_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

    with tempfile.TemporaryDirectory() as f:
        save_path_flexeval_file = Path(f) / "flexeval_file"
        # fmt: off
        result_from_flexeval_file = subprocess.run(
            [  # noqa: S607
                "flexeval_file",
                "--eval_data_loader", "tests.dummy_modules.eval_data_loader.DummyEvalDataLoader",
                "--metrics", "ExactMatch",
                "--save_dir", str(save_path_flexeval_file),
            ],
            check=False,
        )
        # fmt: on
        assert result_from_flexeval_file.returncode == os.EX_OK

        check_if_eval_results_are_correctly_saved(save_path_flexeval_file)


def test_if_flexeval_file_loads_custom_module() -> None:
    custom_metric_code = """\
from flexeval import Metric, MetricResult


class MyCustomMetric(Metric):
    def evaluate(
        self,
        lm_outputs,
        task_inputs_list,
        references_list,
    ) -> MetricResult:
        length_ratios = [
            len(lm_output) / len(references[0])  # Assuming a single reference
            for lm_output, references in zip(lm_outputs, references_list)
        ]

        return MetricResult(
            {"length_ratio": sum(length_ratios) / len(length_ratios)},
            instance_details=[{"length_ratio": ratio} for ratio in length_ratios],
        )
    """
    with tempfile.TemporaryDirectory() as f:
        # prepare the output file from flexeval_lm
        result_from_flexeval_lm = subprocess.run([*GENERATION_CMD, "--save_dir", f], check=False)
        assert result_from_flexeval_lm.returncode == os.EX_OK

        output_file_path = Path(f) / OUTPUTS_FILE_NAME

        # We need to add the temp directory to sys.path so that the custom module can be imported
        os.environ["ADDITIONAL_MODULES_PATH"] = f

        # make a custom module file
        custom_module_path = Path(f) / "my_custom_metric.py"
        custom_module_path.write_text(custom_metric_code)

        save_path_flexeval_file = Path(f) / "flexeval_file"
        # fmt: off
        result_from_flexeval_file = subprocess.run(
            [  # noqa: S607
                "flexeval_file",
                "--eval_file", str(output_file_path),
                "--metrics", "my_custom_metric.MyCustomMetric",
                "--save_dir", str(save_path_flexeval_file),
            ],
            check=False,
        )
        # fmt: on
        assert result_from_flexeval_file.returncode == os.EX_OK

        check_if_eval_results_are_correctly_saved(save_path_flexeval_file)
