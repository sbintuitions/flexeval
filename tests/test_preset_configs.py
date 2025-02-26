import os
from pathlib import Path
from unittest.mock import patch

import datasets
import pytest
from pytest_mock import MockerFixture

from flexeval.core.metric import Metric
from flexeval.core.pairwise_comparison import PairwiseJudge
from flexeval.scripts.flexeval_lm import EvalSetup
from flexeval.utils import instantiate_from_config


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs/EvalSetup").rglob("*.jsonnet")]
    + [str(path) for path in Path("examples/sarashina_evaluation/eval_setups").rglob("*.jsonnet")],
)
def test_if_eval_setup_config_is_valid(config_path: str, mocker: MockerFixture) -> None:
    # datasets.load_dataset takes a long time to load the dataset, so we mock it
    mock_item = {"text": "This is a mock data."}
    mock_dataset = datasets.Dataset.from_list([mock_item for _ in range(5)])
    mocker.patch("datasets.load_dataset", return_value=mock_dataset)
    mocker.patch("jinja2.Template.render", return_value="This is a mock data.")

    eval_setup = instantiate_from_config(config_path)
    assert isinstance(eval_setup, EvalSetup)


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs/Metric").rglob("*.jsonnet")],
)
def test_if_metric_config_is_valid(config_path: str) -> None:
    # we need to set OPENAI_API_KEY to instantiate classes that use OpenAI API
    with patch.dict(os.environ, {"OPENAI_API_KEY": "this-is-a-dummy-key"}):
        eval_setup = instantiate_from_config(config_path)

    assert isinstance(eval_setup, Metric)


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs/PairwiseJudge").rglob("*.jsonnet")],
)
def test_if_pairwise_judge_config_is_valid(config_path: str) -> None:
    # we need to set OPENAI_API_KEY to instantiate classes that use OpenAI API
    with patch.dict(os.environ, {"OPENAI_API_KEY": "this-is-a-dummy-key"}):
        eval_setup = instantiate_from_config(config_path)

    assert isinstance(eval_setup, PairwiseJudge)


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs").rglob("*.jsonnet")],
)
def test_if_config_file_has_comment_at_the_top(config_path: str) -> None:
    with open(config_path) as f:
        first_line = f.readline()
        assert first_line.startswith("/*")


def test_if_there_is_no_duplicates_in_preset_configs() -> None:
    preset_config_files = list(Path("flexeval/preset_configs").rglob("*.jsonnet"))

    seen_file_names = set()
    for file in preset_config_files:
        assert file.name not in seen_file_names, f"{file.name} is duplicated."
        seen_file_names.add(file.name)
