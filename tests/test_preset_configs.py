import os
from pathlib import Path

import datasets
import pytest
from pytest_mock import MockerFixture

from flexeval.core.metric import Metric
from flexeval.core.pairwise_comparison import PairwiseJudge
from flexeval.scripts.flexeval_lm import EvalSetup
from flexeval.utils import instantiate_from_config


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs/EvalSetup").rglob("*.jsonnet")],
)
def test_if_eval_setup_config_is_valid(config_path: str, mocker: MockerFixture) -> None:
    # datasets.load_dataset takes a long time to load the dataset, so we mock it
    possible_keys = ["text", "question", "answer", "summary"]
    mock_item = {k: "This is a mock data." for k in possible_keys}
    possible_label_keys = ["label", "feeling"]
    mock_item.update({k: 0 for k in possible_label_keys})
    mock_dataset = datasets.Dataset.from_list([mock_item for _ in range(5)])
    mocker.patch("datasets.load_dataset", return_value=mock_dataset)
    mocker.patch("datasets.Dataset.filter", return_value=mock_dataset)

    eval_setup = instantiate_from_config(config_path, EvalSetup)
    assert isinstance(eval_setup, EvalSetup)


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs/Metric").rglob("*.jsonnet")],
)
def test_if_metric_config_is_valid(config_path: str) -> None:
    # we need to set OPENAI_API_KEY to instantiate classes that use OpenAI API
    os.environ["OPENAI_API_KEY"] = "this-is-a-dummy-key"

    eval_setup = instantiate_from_config(config_path, Metric)
    assert isinstance(eval_setup, Metric)


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs/PairwiseJudge").rglob("*.jsonnet")],
)
def test_if_pairwise_judge_config_is_valid(config_path: str) -> None:
    # we need to set OPENAI_API_KEY to instantiate classes that use OpenAI API
    os.environ["OPENAI_API_KEY"] = "this-is-a-dummy-key"

    eval_setup = instantiate_from_config(config_path, PairwiseJudge)
    assert isinstance(eval_setup, PairwiseJudge)


@pytest.mark.parametrize(
    "config_path",
    [str(path) for path in Path("flexeval/preset_configs").rglob("*.jsonnet")],
)
def test_if_config_file_has_comment_at_the_top(config_path: str) -> None:
    with open(config_path) as f:
        first_line = f.readline()
        assert first_line.startswith("/*")
