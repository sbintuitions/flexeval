from __future__ import annotations

from typing import Any
from unittest.mock import patch

import datasets
import pytest
from PIL import Image
from pytest_mock import MockerFixture

from flexeval import ChatLLMScore, ChatResponse
from flexeval.core.language_model import LanguageModel
from flexeval.core.language_model.base import LMOutput
from flexeval.utils import instantiate_from_config
from tests.dummy_modules import DummyVLM

VISION_PRESETS = [
    "flexeval/preset_configs/EvalSetup/en_vision/mmmu.jsonnet",
    "flexeval/preset_configs/EvalSetup/ja_vision/jmmmu.jsonnet",
]


def _mmmu_like_mock_dataset() -> datasets.Dataset:
    # One mock shape covers both presets: JMMMU rows are structurally identical
    # to MMMU rows (image_1..image_7, stringified `options`, `answer`,
    # `question_type`); JMMMU only adds per-image license/attribution columns,
    # which the configs never read. Hence no separate JMMMU-shaped mock.
    image = Image.new("RGB", (64, 48), color=(73, 109, 137))
    row = {
        "id": "validation_Art_1",
        "question": "<image 1> Who painted this?",
        "options": "['Alice', 'Bob', 'Carol', 'Dave']",
        "answer": "B",
        "question_type": "multiple-choice",
        "image_1": image,
        "image_2": image,
        "image_3": None,
        "image_4": None,
        "image_5": None,
        "image_6": None,
        "image_7": None,
    }
    open_row = {**row, "id": "validation_Art_2", "question_type": "open", "options": "[]"}
    return datasets.Dataset.from_list([row, open_row])


@pytest.mark.parametrize("config_path", VISION_PRESETS)
def test_vision_preset_end_to_end_with_dummy_vlm(config_path: str, mocker: MockerFixture) -> None:
    load_dataset_mock = mocker.patch("datasets.load_dataset", return_value=_mmmu_like_mock_dataset())

    eval_setup = instantiate_from_config(config_path)
    assert isinstance(eval_setup, ChatResponse)

    # all subjects (subsets) are loaded and concatenated
    num_subjects = load_dataset_mock.call_count
    assert num_subjects >= 28

    dataset = eval_setup.eval_dataset
    # the open-ended rows are filtered out by keep_conditions
    assert len(dataset) == num_subjects

    instance = dataset[0]
    # concatenated subsets record their subject name for category-wise scores
    assert instance.extra_info["subset"] == load_dataset_mock.call_args_list[0].kwargs["name"]

    content = instance.messages[0]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    text_parts = [part for part in content if part["type"] == "text"]
    assert len(image_parts) == 2  # image_3..7 are None and skipped
    assert len(text_parts) == 1
    assert "<image>" in text_parts[0]["text"]  # <image 1> tag normalized
    assert "A. Alice" in text_parts[0]["text"]  # options formatted
    assert instance.references == ["B"]

    # The DummyVLM decodes the base64 images, proving the pipeline is well-formed.
    lm_output = DummyVLM().generate_chat_response(instance.messages)
    assert isinstance(lm_output, LMOutput)
    assert lm_output.text == "2 image(s) and 1 text part(s) are given!"

    # The preset's last_choice_exact_match metric scores a letter answer as correct.
    assert isinstance(eval_setup.metrics, list)
    metric = eval_setup.metrics[2]  # ExactMatch with LastChoiceExtractor
    result = metric.evaluate(
        lm_outputs=["The answer is B."],
        references_list=[instance.references],
        extra_info_list=[instance.extra_info],
    )
    assert result.summary["last_choice_exact_match"] == pytest.approx(1.0)


HERON_EVAL_SETUP_PATH = "flexeval/preset_configs/EvalSetup/ja_vision/heron_bench.jsonnet"
HERON_JUDGE_PATH = "flexeval/preset_configs/Metric/heron_bench_judge.jsonnet"


def _heron_like_mock_dataset() -> datasets.Dataset:
    # The source images are JPEG (see the preset's image-fidelity note).
    image = Image.new("RGB", (64, 48), color=(73, 109, 137))
    return datasets.Dataset.from_list(
        [
            {
                "question_id": 0,
                "image": image,
                "category": "complex",
                "image_category": "anime",
                "context": "アニメ映画のワンシーンです。",
                "text": "この作品の監督は誰ですか？",
                "answer": {"gpt-4-0125-preview": "宮崎駿監督の作品だと考えられます。"},
            },
        ],
    )


def test_heron_bench_preset_end_to_end_with_dummy_vlm(mocker: MockerFixture) -> None:
    mocker.patch("datasets.load_dataset", return_value=_heron_like_mock_dataset())

    eval_setup = instantiate_from_config(HERON_EVAL_SETUP_PATH)
    assert isinstance(eval_setup, ChatResponse)

    instance = eval_setup.eval_dataset[0]
    content = instance.messages[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["text"] == "この作品の監督は誰ですか？"
    assert instance.references == ["宮崎駿監督の作品だと考えられます。"]

    lm_output = DummyVLM().generate_chat_response(instance.messages)
    assert isinstance(lm_output, LMOutput)
    assert lm_output.text == "1 image(s) and 1 text part(s) are given!"


class _CannedJudge(LanguageModel):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.queried_inputs: list[list[dict[str, Any]]] = []

    def _batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, Any]]],
        tools_list: list[list[dict[str, Any]] | None] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[LMOutput]:
        self.queried_inputs.extend(chat_messages_list)
        return [LMOutput(text=self.responses.pop(0), finish_reason="stop") for _ in chat_messages_list]


def test_heron_bench_judge_scores_with_canned_judge() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "this-is-a-dummy-key"}):
        metric = instantiate_from_config(HERON_JUDGE_PATH)
    assert isinstance(metric, ChatLLMScore)
    metric.language_model = _CannedJudge(["3", "5"])
    metric.disable_tqdm = True

    result = metric.evaluate(
        lm_outputs=["ジブリ作品です。", "宮崎駿監督の作品です。"],
        references_list=[["宮崎駿監督の作品だと考えられます。"]] * 2,
        extra_info_list=[
            {"context": "アニメ映画のワンシーンです。", "text": "この作品の監督は誰ですか？", "category": "complex"},
            {"context": "アニメ映画のワンシーンです。", "text": "この作品の監督は誰ですか？", "category": "detail"},
        ],
    )
    assert result.summary["llm_score"] == pytest.approx(4.0)
    assert result.summary["llm_score/category/complex"] == pytest.approx(3.0)
    assert result.summary["llm_score/category/detail"] == pytest.approx(5.0)

    # The judge prompt embeds context, question, reference, and prediction.
    judge_prompt = metric.language_model.queried_inputs[0][-1]["content"]
    fragments = (
        "アニメ映画のワンシーン",
        "この作品の監督は誰ですか",
        "宮崎駿監督の作品だと考えられます",
        "ジブリ作品です",
    )
    for fragment in fragments:
        assert fragment in judge_prompt
