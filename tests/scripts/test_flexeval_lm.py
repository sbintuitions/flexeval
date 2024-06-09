from __future__ import annotations

import json
import os
import subprocess
import tempfile
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from typing import Any

import pytest

from flexeval.scripts.common import CONFIG_FILE_NAME, METRIC_FILE_NAME, OUTPUTS_FILE_NAME

# fmt: off
CHAT_RESPONSE_CMD = [
    "flexeval_lm",
    "--language_model", "tests.dummy_modules.DummyLanguageModel",
    "--eval_setup", "ChatResponse",
    "--eval_setup.eval_dataset", "tests.dummy_modules.DummyChatDataset",
    "--eval_setup.metrics+=ExactMatch",
    "--eval_setup.gen_kwargs", "{}",
    "--eval_setup.batch_size", "1",
]

GENERATION_CMD = [
    "flexeval_lm",
    "--language_model", "tests.dummy_modules.DummyLanguageModel",
    "--eval_setup", "Generation",
    "--eval_setup.eval_dataset", "tests.dummy_modules.DummyGenerationDataset",
    "--eval_setup.prompt_template", "Jinja2PromptTemplate",
    "--eval_setup.prompt_template.template", "{{text}}",
    "--eval_setup.metrics+=ExactMatch",
    "--eval_setup.gen_kwargs", "{}",
    "--eval_setup.batch_size", "1",
]

MULTIPLE_CHOICE_CMD = [
    "flexeval_lm",
    "--language_model", "tests.dummy_modules.DummyLanguageModel",
    "--eval_setup", "MultipleChoice",
    "--eval_setup.eval_dataset", "tests.dummy_modules.DummyMultipleChoiceDataset",
    "--eval_setup.prompt_template", "Jinja2PromptTemplate",
    "--eval_setup.prompt_template.template", "{{text}}",
    "--eval_setup.batch_size", "1",
]

PERPLEXITY_CMD = [
    "flexeval_lm",
    "--language_model", "tests.dummy_modules.DummyLanguageModel",
    "--eval_setup", "Perplexity",
    "--eval_setup.eval_dataset", "tests.dummy_modules.DummyTextDataset",
    "--eval_setup.batch_size", "1",
]
# fmt: on


def read_jsonl(path: str | PathLike[str]) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def check_if_eval_results_are_correctly_saved(save_dir: str | PathLike[str], no_outputs: bool = False) -> None:
    with open(next(Path(save_dir).rglob(CONFIG_FILE_NAME))) as f_json:
        saved_config = json.load(f_json)
    assert saved_config["metadata"]["flexeval_version"] == version("flexeval")

    with open(next(Path(save_dir).rglob(METRIC_FILE_NAME))) as f_json:
        assert json.load(f_json)

    if not no_outputs:
        assert read_jsonl(next(Path(save_dir).rglob(OUTPUTS_FILE_NAME)))


@pytest.mark.parametrize(
    "command",
    [CHAT_RESPONSE_CMD, GENERATION_CMD, MULTIPLE_CHOICE_CMD, PERPLEXITY_CMD],
)
def test_cli(command: list[str]) -> None:
    with tempfile.TemporaryDirectory() as f:
        result = subprocess.run([*command, "--save_dir", f], check=False)
        assert result.returncode == 0

        check_if_eval_results_are_correctly_saved(f, no_outputs="Perplexity" in command)


def test_evaluate_suite_cli() -> None:
    with tempfile.TemporaryDirectory() as f:
        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            "--eval_setup", "tests/dummy_modules/configs/eval_suite.jsonnet",
            "--save_dir", f,
        ]
        # fmt: on
        result = subprocess.run(command, check=False)
        assert result.returncode == 0

        for task_name in ["generation", "multiple_choice", "perplexity"]:
            check_if_eval_results_are_correctly_saved(Path(f) / task_name, no_outputs=task_name == "perplexity")

            # check if the saved config is reusable
            command = ["flexeval_lm", "--config", str(Path(f) / task_name / CONFIG_FILE_NAME)]
            result = subprocess.run(command, check=False)
            assert result.returncode == 0


@pytest.mark.parametrize(
    "eval_setup_args",
    [
        ["--eval_setup", "generation"],
        ["--eval_setup", "multiple_choice"],
        ["--eval_setup.gen", "generation", "--eval_setup.multi", "multiple_choice"],
        [
            "--eval_setup.gen",
            "generation",
            "--eval_setup.multi",
            str(Path(__file__).parent.parent / "dummy_modules" / "configs" / "multiple_choice.jsonnet"),
        ],
        # with overrides
        ["--eval_setup", "generation", "--eval_setup.batch_size", "8"],
        ["--eval_setup", "generation", "--eval_setup.gen_kwargs.do_sample", "true"],
        ["--eval_setup.gen", "generation", "--eval_setup.gen.batch_size", "8", "--eval_setup.multi", "multiple_choice"],
    ],
)
def test_flexeval_lm_with_preset_config(eval_setup_args: list[str]) -> None:
    with tempfile.TemporaryDirectory() as f:
        os.environ["PRESET_CONFIG_EVAL_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            *eval_setup_args,
            "--save_dir", f,
        ]
        # fmt: on

        result = subprocess.run(command, check=False)
        assert result.returncode == 0

        check_if_eval_results_are_correctly_saved(f)


@pytest.mark.parametrize(
    ("batch_size", "do_sample"),
    [(8, True), (4, False)],
)
def test_if_flexeval_lm_with_preset_correctly_overridden(batch_size: int, do_sample: bool) -> None:
    os.environ["PRESET_CONFIG_EVAL_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

    # first run without overrides
    # the result will be compared to the one with overrides
    with tempfile.TemporaryDirectory() as f:
        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            "--eval_setup", "generation",
            "--save_dir", f,
        ]
        # fmt: on
        result = subprocess.run(command, check=False)
        assert result.returncode == 0

        check_if_eval_results_are_correctly_saved(f)

        outputs_without_overrides = read_jsonl(next(Path(f).rglob(OUTPUTS_FILE_NAME)))

    # run with overrides
    with tempfile.TemporaryDirectory() as f:
        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            "--eval_setup", "generation",
            "--eval_setup.batch_size", str(batch_size),
            "--eval_setup.gen_kwargs.do_sample", str(do_sample),
            "--save_dir", f,
        ]
        # fmt: on

        result = subprocess.run(command, check=False)
        assert result.returncode == 0

        check_if_eval_results_are_correctly_saved(f)

        with open(next(Path(f).rglob(CONFIG_FILE_NAME))) as f_json:
            saved_config = json.load(f_json)
        assert saved_config["eval_setup"]["init_args"]["batch_size"] == batch_size
        assert saved_config["eval_setup"]["init_args"]["gen_kwargs"]["do_sample"] == do_sample

        outputs_with_overrides = read_jsonl(next(Path(f).rglob(OUTPUTS_FILE_NAME)))
        assert outputs_with_overrides != outputs_without_overrides


@pytest.mark.parametrize(
    "command",
    [CHAT_RESPONSE_CMD, GENERATION_CMD, MULTIPLE_CHOICE_CMD, PERPLEXITY_CMD],
)
def test_if_cli_raises_error_when_save_data_exists(command: list[str]) -> None:
    with tempfile.TemporaryDirectory() as f:
        config_file = Path(f) / CONFIG_FILE_NAME
        config_file.touch()
        result = subprocess.run([*command, "--save_dir", f], check=False)
        assert result.returncode == 0
        # check if config_file is not modified and empty
        assert config_file.read_text() == ""


@pytest.mark.parametrize(
    "command",
    [CHAT_RESPONSE_CMD, GENERATION_CMD, MULTIPLE_CHOICE_CMD, PERPLEXITY_CMD],
)
def test_if_saved_config_can_be_reused_to_run_eval(command: list[str]) -> None:
    with tempfile.TemporaryDirectory() as f:
        result = subprocess.run([*command, "--save_dir", f], check=False)
        assert result.returncode == 0
        check_if_eval_results_are_correctly_saved(f, no_outputs="Perplexity" in command)

        saved_config_file_path = str(Path(f) / CONFIG_FILE_NAME)
        with open(saved_config_file_path) as f_json:
            assert json.load(f_json)

        new_save_path = str(Path(f) / "new_save_dir")
        # fmt: off
        new_command = [
            "flexeval_lm",
            "--config", saved_config_file_path,
            "--save_dir", new_save_path,
        ]
        # fmt: on
        result = subprocess.run(new_command, check=False)
        assert result.returncode == 0
        check_if_eval_results_are_correctly_saved(new_save_path, no_outputs="Perplexity" in command)

        new_saved_config_file_path = str(Path(new_save_path) / CONFIG_FILE_NAME)

        # check if the contents of the saved config are the same
        with open(saved_config_file_path) as f_json:
            saved_config = json.load(f_json)
        with open(new_saved_config_file_path) as f_json:
            new_saved_config = json.load(f_json)
        for key in saved_config:
            if key in {"save_dir", "config"}:
                continue
            assert saved_config[key] == new_saved_config[key]
