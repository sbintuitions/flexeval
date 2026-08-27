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

from flexeval import Generation, Perplexity, RandomFewShotGenerator
from flexeval.core.result_recorder.local_recorder import CONFIG_FILE_NAME, METRIC_FILE_NAME, OUTPUTS_FILE_NAME
from flexeval.scripts.flexeval_lm import generate_eval_entries, maybe_replace_random_seed
from tests.dummy_modules import DummyGenerationDataset, DummyTextDataset

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
        assert result.returncode == os.EX_OK

        check_if_eval_results_are_correctly_saved(f, no_outputs="Perplexity" in command)


def test_evaluate_suite_cli() -> None:
    with tempfile.TemporaryDirectory() as f:
        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            "--eval_setups", "tests/dummy_modules/configs/eval_suite.jsonnet",
            "--save_dir", f,
        ]
        # fmt: on
        result = subprocess.run(command, check=False)
        assert result.returncode == os.EX_OK

        for task_name in ["generation", "multiple_choice", "perplexity"]:
            check_if_eval_results_are_correctly_saved(Path(f) / task_name, no_outputs=task_name == "perplexity")

            # check if the saved config is reusable
            command = ["flexeval_lm", "--config", str(Path(f) / task_name / CONFIG_FILE_NAME)]
            result = subprocess.run(command, check=False)
            assert result.returncode == os.EX_OK


@pytest.mark.parametrize(
    "eval_setup_args",
    [
        ["--eval_setup", "generation"],
        ["--eval_setup", "multiple_choice"],
        ["--eval_setups.gen", "generation", "--eval_setups.multi", "multiple_choice"],
        [
            "--eval_setups.gen",
            "generation",
            "--eval_setups.multi",
            str(Path(__file__).parent.parent / "dummy_modules" / "configs" / "multiple_choice.jsonnet"),
        ],
        # with overrides
        ["--eval_setup", "generation", "--eval_setup.batch_size", "8"],
        ["--eval_setup", "generation", "--eval_setup.gen_kwargs.do_sample", "true"],
        [
            "--eval_setups.gen",
            "generation",
            "--eval_setups.gen.batch_size",
            "8",
            "--eval_setups.multi",
            "multiple_choice",
        ],
    ],
)
def test_flexeval_lm_with_preset_config(eval_setup_args: list[str]) -> None:
    with tempfile.TemporaryDirectory() as f:
        os.environ["PRESET_CONFIG_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            *eval_setup_args,
            "--save_dir", f,
        ]
        # fmt: on

        result = subprocess.run(command, check=False)
        assert result.returncode == os.EX_OK

        check_if_eval_results_are_correctly_saved(f)


@pytest.mark.parametrize(
    ("batch_size", "do_sample", "is_eval_setups"),
    [(8, True, True), (4, False, False)],
)
def test_if_flexeval_lm_with_preset_correctly_overridden(
    batch_size: int,
    do_sample: bool,
    is_eval_setups: bool,
) -> None:
    os.environ["PRESET_CONFIG_DIR"] = str(Path(__file__).parent.parent / "dummy_modules" / "configs")

    if is_eval_setups:
        eval_setup_arg = "eval_setups.gen"
        save_folder_name = "gen"
    else:
        eval_setup_arg = "eval_setup"
        save_folder_name = None

    # first run without overrides
    # the result will be compared to the one with overrides
    with tempfile.TemporaryDirectory() as f:
        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            f"--{eval_setup_arg}", "generation",
            "--save_dir", f,
        ]
        # fmt: on
        result = subprocess.run(command, check=False)
        assert result.returncode == os.EX_OK

        save_dir = Path(f)
        if save_folder_name:
            save_dir = Path(f) / save_folder_name

        check_if_eval_results_are_correctly_saved(save_dir)

        outputs_without_overrides = read_jsonl(next(Path(save_dir).rglob(OUTPUTS_FILE_NAME)))

    # run with overrides
    with tempfile.TemporaryDirectory() as f:
        # fmt: off
        command = [
            "flexeval_lm",
            "--language_model", "tests.dummy_modules.DummyLanguageModel",
            f"--{eval_setup_arg}", "generation",
            f"--{eval_setup_arg}.batch_size", str(batch_size),
            f"--{eval_setup_arg}.gen_kwargs.do_sample", str(do_sample),
            "--save_dir", f,
        ]
        # fmt: on

        result = subprocess.run(command, check=False)
        assert result.returncode == os.EX_OK

        save_dir = Path(f)
        if save_folder_name:
            save_dir = Path(f) / save_folder_name

        check_if_eval_results_are_correctly_saved(save_dir)

        with open(next(Path(save_dir).rglob(CONFIG_FILE_NAME))) as f_json:
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
        assert result.returncode == os.EX_DATAERR
        # check if config_file is not modified and empty
        assert config_file.read_text() == ""


@pytest.mark.parametrize(
    "command",
    [CHAT_RESPONSE_CMD, GENERATION_CMD, MULTIPLE_CHOICE_CMD, PERPLEXITY_CMD],
)
def test_if_saved_config_can_be_reused_to_run_eval(command: list[str]) -> None:
    with tempfile.TemporaryDirectory() as f:
        result = subprocess.run([*command, "--save_dir", f], check=False)
        assert result.returncode == os.EX_OK
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
        assert result.returncode == os.EX_OK
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


def test_if_flexeval_lm_loads_custom_module() -> None:
    custom_metric_code = """\
from flexeval import Metric, MetricResult
from flexeval.core.metric.utils import extract_text_from_outputs

class MyCustomMetric(Metric):
    def evaluate(
        self,
        lm_outputs,
        extra_info_list,
        references_list,
    ) -> MetricResult:
        lm_outputs = extract_text_from_outputs(lm_outputs)
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
        # We need to add the temp directory to sys.path so that the custom module can be imported
        os.environ["ADDITIONAL_MODULES_PATH"] = f

        # make a custom module file
        custom_module_dir = Path(f) / "custom_modules"
        custom_module_dir.mkdir()
        # make __init__.py
        custom_module_dir.joinpath("__init__.py").touch()
        custom_module_path = custom_module_dir / "my_custom_metric.py"
        custom_module_path.write_text(custom_metric_code)

        command = [
            *GENERATION_CMD,
            "--eval_setup.metrics+=custom_modules.my_custom_metric.MyCustomMetric",
            "--save_dir",
            f,
        ]

        result = subprocess.run(command, check=False)
        assert result.returncode == os.EX_OK

        check_if_eval_results_are_correctly_saved(f)


@pytest.fixture
def mock_eval_data() -> dict:
    return {"setup": "dummy_setup_object", "config": {"task": "test", "metric": "acc"}, "group": "test_group"}


def test_no_repeat_with_group(mock_eval_data: dict) -> None:
    result = generate_eval_entries(
        eval_setup=mock_eval_data["setup"],
        config_content=mock_eval_data["config"],
        group=mock_eval_data["group"],
        num_repeats=1,
    )
    expected = [("dummy_setup_object", {"task": "test", "metric": "acc"}, "test_group")]
    assert result == expected
    assert len(result) == 1


def test_no_repeat_no_group(mock_eval_data: dict) -> None:
    result = generate_eval_entries(
        eval_setup=mock_eval_data["setup"], config_content=mock_eval_data["config"], group=None, num_repeats=1
    )
    expected = [("dummy_setup_object", {"task": "test", "metric": "acc"}, None)]
    assert result == expected
    assert len(result) == 1


def test_multiple_repeats_with_group(mock_eval_data: dict) -> None:
    num_repeats = 3
    result = generate_eval_entries(
        eval_setup=mock_eval_data["setup"],
        config_content=mock_eval_data["config"],
        group=mock_eval_data["group"],
        num_repeats=num_repeats,
    )
    expected_metadatas = ["test_group/run0", "test_group/run1", "test_group/run2"]

    assert len(result) == num_repeats
    for i, entry in enumerate(result):
        assert entry[0] == mock_eval_data["setup"]
        assert entry[1] == mock_eval_data["config"]
        assert entry[2] == expected_metadatas[i]


def test_non_positive_num_repeats(mock_eval_data: dict) -> None:
    with pytest.raises(ValueError):
        generate_eval_entries(
            eval_setup=mock_eval_data["setup"],
            config_content=mock_eval_data["config"],
            group=mock_eval_data["group"],
            num_repeats=-1,
        )


def test_maybe_replace_random_seed() -> None:
    # eval setup w/ random seed: should update random seed
    generation_eval_setup = Generation(
        eval_dataset=DummyGenerationDataset(), gen_kwargs={}, prompt_template="dummy", random_seed=42
    )
    generation_config = {"init_args": {"random_seed": 42}}
    new_generation_eval_setup, new_generation_config = maybe_replace_random_seed(
        generation_eval_setup, generation_config, seed_increment=1
    )
    assert new_generation_eval_setup.random_seed == 43
    assert new_generation_config["init_args"]["random_seed"] == 43

    # eval setup w/o random seed: should not change anything
    perplexity_eval_setup = Perplexity(eval_dataset=DummyTextDataset())
    perplexity_config = {}
    new_perplexity_eval_setup, new_perplexity_config = maybe_replace_random_seed(
        perplexity_eval_setup, perplexity_config, seed_increment=1
    )
    assert id(new_perplexity_config) == id(perplexity_config)
    assert id(new_perplexity_eval_setup) == id(perplexity_eval_setup)


def test_maybe_replace_random_seed_replaces_few_shot_generator_with_new_instance() -> None:
    dataset = DummyGenerationDataset()
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=1, seed=42)
    generation_eval_setup = Generation(
        eval_dataset=dataset,
        gen_kwargs={},
        prompt_template="dummy",
        few_shot_generator=few_shot_generator,
        random_seed=42,
    )
    generation_config = {
        "init_args": {
            "random_seed": 42,
            "few_shot_generator": {
                "class_path": "flexeval.RandomFewShotGenerator",
                "init_args": {"dataset": "dummy", "num_shots": 1, "seed": 42, "num_trials_to_avoid_leak": 3},
            },
        },
    }

    new_eval_setup, new_config = maybe_replace_random_seed(generation_eval_setup, generation_config, seed_increment=1)

    # a new, distinct few_shot_generator instance is used, but the dataset is shared (not deep-copied)
    assert new_eval_setup.few_shot_generator is not few_shot_generator
    assert new_eval_setup.few_shot_generator.dataset is dataset

    # the derived generator's samples are reproducible from seed alone,
    # matching an independently constructed generator with the incremented seed
    directly_constructed = RandomFewShotGenerator(dataset=dataset, num_shots=1, seed=43)
    assert new_eval_setup.few_shot_generator() == directly_constructed()

    # the config's few_shot_generator seed is incremented too, for reproducibility
    assert new_config["init_args"]["few_shot_generator"]["init_args"]["seed"] == 43
    # the original config is untouched
    assert generation_config["init_args"]["few_shot_generator"]["init_args"]["seed"] == 42


def test_maybe_replace_random_seed_without_few_shot_generator_seed_key_in_config_is_defensive() -> None:
    # if the config's few_shot_generator init_args has no "seed" key, nothing should be touched there
    dataset = DummyGenerationDataset()
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=1, seed=42)
    generation_eval_setup = Generation(
        eval_dataset=dataset,
        gen_kwargs={},
        prompt_template="dummy",
        few_shot_generator=few_shot_generator,
        random_seed=42,
    )
    generation_config = {
        "init_args": {
            "random_seed": 42,
            "few_shot_generator": {
                "class_path": "flexeval.RandomFewShotGenerator",
                "init_args": {"dataset": "dummy", "num_shots": 1, "num_trials_to_avoid_leak": 3},
            },
        },
    }

    new_eval_setup, new_config = maybe_replace_random_seed(generation_eval_setup, generation_config, seed_increment=1)

    assert new_eval_setup.few_shot_generator is not few_shot_generator
    assert "seed" not in new_config["init_args"]["few_shot_generator"]["init_args"]


def test_maybe_replace_random_seed_with_none_few_shot_generator_does_not_error() -> None:
    dataset = DummyGenerationDataset()
    generation_eval_setup = Generation(
        eval_dataset=dataset,
        gen_kwargs={},
        prompt_template="dummy",
        few_shot_generator=None,
        random_seed=42,
    )
    generation_config = {"init_args": {"random_seed": 42}}

    new_eval_setup, new_config = maybe_replace_random_seed(generation_eval_setup, generation_config, seed_increment=1)

    assert new_eval_setup.few_shot_generator is None
    assert new_eval_setup.random_seed == 43
    assert new_config["init_args"]["random_seed"] == 43


def test_generate_eval_entries_gives_each_repeat_a_distinct_few_shot_generator() -> None:
    dataset = DummyGenerationDataset()
    few_shot_generator = RandomFewShotGenerator(dataset=dataset, num_shots=1, seed=42)
    generation_eval_setup = Generation(
        eval_dataset=dataset,
        gen_kwargs={},
        prompt_template="dummy",
        few_shot_generator=few_shot_generator,
        random_seed=42,
    )
    generation_config = {
        "init_args": {
            "random_seed": 42,
            "few_shot_generator": {
                "class_path": "flexeval.RandomFewShotGenerator",
                "init_args": {"dataset": "dummy", "num_shots": 1, "seed": 42, "num_trials_to_avoid_leak": 3},
            },
        },
    }

    entries = generate_eval_entries(
        eval_setup=generation_eval_setup,
        config_content=generation_config,
        group="test_group",
        num_repeats=3,
    )

    generators = [entry[0].few_shot_generator for entry in entries]
    # all generators must be distinct objects from each other and from the original
    assert len({id(g) for g in generators}) == len(generators)
    assert few_shot_generator not in generators

    # run0 (seed_increment=0) must be reproducible from the original seed alone,
    # regardless of the fact that it's derived via with_seed_increment(0)
    directly_constructed = RandomFewShotGenerator(dataset=dataset, num_shots=1, seed=42)
    assert generators[0]() == directly_constructed()


def test_generate_eval_entries_with_perplexity_setup_does_not_error() -> None:
    # Perplexity has neither `random_seed` nor `few_shot_generator`
    perplexity_eval_setup = Perplexity(eval_dataset=DummyTextDataset())
    perplexity_config = {"init_args": {}}

    entries = generate_eval_entries(
        eval_setup=perplexity_eval_setup,
        config_content=perplexity_config,
        group="test_group",
        num_repeats=3,
    )
    assert len(entries) == 3
    for entry in entries:
        assert entry[0] is perplexity_eval_setup


@pytest.mark.parametrize(
    "num_repeats",
    [1, 3, 5],
)
def test_flexeval_lm_with_num_repeats(num_repeats: int) -> None:
    with tempfile.TemporaryDirectory() as f:
        # fmt: off
        command = [*CHAT_RESPONSE_CMD, "--num_repeats", str(num_repeats), "--save_dir", f]
        # fmt: on

        result = subprocess.run(command, check=False)
        assert result.returncode == os.EX_OK

        if num_repeats == 1:
            check_if_eval_results_are_correctly_saved(f)
        else:
            for i in range(num_repeats):
                check_if_eval_results_are_correctly_saved(f"{f}/run{i}")


@pytest.mark.parametrize("force", [True, False])
@pytest.mark.parametrize("retry", [True, False])
def test_flexeval_lm_with_retry(retry: bool, force: bool) -> None:
    with tempfile.TemporaryDirectory() as f:
        num_repeats = 3
        # First run to create initial results
        command = [*CHAT_RESPONSE_CMD, "--num_repeats", str(num_repeats), "--save_dir", f]
        result = subprocess.run(command, check=False)
        assert result.returncode == os.EX_OK
        for i in range(num_repeats):
            check_if_eval_results_are_correctly_saved(f"{f}/run{i}")

        # Delete metrics.json and outputs.json in run1 to simulate a failed run
        os.remove(f"{f}/run1/{METRIC_FILE_NAME}")  # noqa: PTH107
        os.remove(f"{f}/run1/{OUTPUTS_FILE_NAME}")  # noqa: PTH107

        # Second run with resume
        command = [
            *CHAT_RESPONSE_CMD,
            "--num_repeats",
            str(num_repeats),
            "--save_dir",
            f,
            "--retry",
            str(retry).lower(),
            "--force",
            str(force).lower(),
        ]

        result = subprocess.run(command, check=False, capture_output=True)

        if force:
            assert "Skipping the evaluation for group: run0" not in result.stderr.decode()
            assert "Skipping the evaluation for group: run2" not in result.stderr.decode()
            check_if_eval_results_are_correctly_saved(f"{f}/run1")
        elif retry:
            assert "Skipping the evaluation for group: run0" in result.stderr.decode()
            assert "Skipping the evaluation for group: run2" in result.stderr.decode()
            check_if_eval_results_are_correctly_saved(f"{f}/run1")
        elif not retry and not force:
            assert not Path(f"{f}/run1/{METRIC_FILE_NAME}").exists()
            assert not Path(f"{f}/run1/{OUTPUTS_FILE_NAME}").exists()
