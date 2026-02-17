from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jinja2 import Template

from flexeval.core.chat_dataset import HFChatDataset, JsonlChatDataset, TemplateChatDataset, load_jinja2_template

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the Web for specified query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "str - query for search"}},
            "required": ["query"],
        },
        "return": {"type": "string", "description": "snippets: list"},
    },
}

DATASETS_TO_TEST = [
    (
        HFChatDataset,
        {
            "path": "tests/dummy_modules/hf_dataset",
            "split": "train",
        },
        False,
    ),
    (
        JsonlChatDataset,
        {
            "path": "tests/dummy_modules/test.jsonl",
        },
        False,
    ),
    (
        JsonlChatDataset,
        {
            "path": "tests/dummy_modules/test_with_tools.jsonl",
        },
        True,
    ),
    (
        JsonlChatDataset,
        {
            "path": "tests/dummy_modules/test.jsonl",
            "tools": [TOOL_DEFINITION],
        },
        True,
    ),
]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs", "has_tools"),
    DATASETS_TO_TEST,
)
def test_template_dataset_with_reference(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
    has_tools: bool,
) -> None:
    system_message = "You are a quiz player."
    chat_dataset = dataset_class(
        **kwargs,
        input_template="{{ question }}",
        reference_template="{{ answers[0] }}",
        extra_info_templates={"question_as_extra_info": "{{ question }}"},
        system_message_template=system_message,
    )

    assert len(chat_dataset) == 10

    assert chat_dataset[0].messages == [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": "What is the highest mountain in the world.",
        },
    ]
    assert chat_dataset[0].references == ["Mount Everest"]
    assert chat_dataset[0].extra_info["question"] == "What is the highest mountain in the world."
    if has_tools:
        assert chat_dataset[0].tools == [TOOL_DEFINITION]
    else:
        assert chat_dataset[0].tools is None


@pytest.mark.parametrize(
    ("dataset_class", "kwargs", "has_tools"),
    DATASETS_TO_TEST,
)
def test_template_dataset_with_reference_list(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
    has_tools: bool,
) -> None:
    system_message = "You are a quiz player."
    chat_dataset = dataset_class(
        **kwargs,
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
        extra_info_templates={"question_as_extra_info": "{{ question }}"},
        system_message_template=system_message,
    )

    assert len(chat_dataset) == 10

    assert chat_dataset[0].messages == [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": "What is the highest mountain in the world.",
        },
    ]
    assert chat_dataset[0].references == ["Mount Everest", "Everest"]
    assert chat_dataset[0].extra_info["question"] == "What is the highest mountain in the world."
    if has_tools:
        assert chat_dataset[0].tools == [TOOL_DEFINITION]
    else:
        assert chat_dataset[0].tools is None


@pytest.mark.parametrize(
    ("dataset_class", "kwargs", "has_tools"),
    DATASETS_TO_TEST,
)
def test_data_range(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
    has_tools: bool,  # noqa: ARG001
) -> None:
    data_range = (2, 5)
    dataset = dataset_class(
        **kwargs,
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
        data_range=data_range,
    )
    assert list(range(*data_range)) == [i.extra_info["id"] for i in dataset]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs", "has_tools"),
    DATASETS_TO_TEST,
)
def test_keep_conditions(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
    has_tools: bool,  # noqa: ARG001
) -> None:
    original_dataset = dataset_class(
        **kwargs,
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
    )

    filtered_dataset = dataset_class(
        **kwargs,
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
        keep_conditions={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.references) == 1


@pytest.mark.parametrize(
    ("dataset_class", "kwargs", "has_tools"),
    DATASETS_TO_TEST,
)
def test_remove_conditions(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
    has_tools: bool,  # noqa: ARG001
) -> None:
    original_dataset = dataset_class(
        **kwargs,
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
    )

    filtered_dataset = dataset_class(
        **kwargs,
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
        remove_conditions={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.references) > 1


@pytest.fixture
def dummy_template_file(tmp_path: Path) -> Path:
    template_content = "Hello {{ name }}!"
    template_file = tmp_path / "dummy.j2"
    template_file.write_text(template_content, encoding="utf-8")
    return template_file


def test_load_jinja2_template(dummy_template_file: Path) -> None:
    template_from_path = load_jinja2_template(dummy_template_file)
    embed_result = template_from_path.render(name="flexeval")
    assert isinstance(template_from_path, Template)
    assert embed_result == "Hello flexeval!"

    template_from_string = load_jinja2_template("Hello {{ name }}!")
    embed_result = template_from_string.render(name="flexeval")
    assert isinstance(template_from_string, Template)
    assert embed_result == "Hello flexeval!"

    template_from_string = load_jinja2_template("a" * 1000)
    embed_result = template_from_string.render()
    assert isinstance(template_from_string, Template)
    assert embed_result == "a" * 1000


@pytest.mark.parametrize(
    "parse_input_utterance",
    ["literal_eval", "json_loads", None],
)
def test_parse_input_utterance(parse_input_utterance: str) -> None:
    input_template = """[
        {"type": "image_url", "image_url": {"url": "{{ image }}"}},
        {"type": "text", "text": "{{ question }}"}
    ]"""
    dataset = TemplateChatDataset(
        items=[
            {
                "question": "Describe the color of this object.",
                "answer": "red",
                "image": "http://example.com/image1.jpg",
            },
        ],
        input_template=input_template,
        parse_input_utterance=parse_input_utterance,
    )

    input_utterance = dataset[0].messages[0]["content"]

    if parse_input_utterance is None:
        assert isinstance(input_utterance, str)

    else:
        assert isinstance(input_utterance, list)
        assert input_utterance[0]["type"] == "image_url"
        assert input_utterance[0]["image_url"]["url"] == "http://example.com/image1.jpg"
        assert input_utterance[1]["type"] == "text"
        assert input_utterance[1]["text"] == "Describe the color of this object."


def test_preprocessors() -> None:
    from flexeval.core.chat_dataset.template_based import Preprocessor

    class ToBase64(Preprocessor):
        def __call__(self, item: dict) -> dict:
            image = item["image"]  # noqa: F841 # simulate using the image for conversion
            item["image_base64"] = "data:image/jpeg;base64,..."
            return item

    input_template = "{{ image_base64 }}"
    dataset = TemplateChatDataset(
        items=[
            {
                "question": "Describe the color of this object.",
                "answer": "red",
                "image": "http://example.com/image1.jpg",
            },
        ],
        input_template=input_template,
        preprocessors=[ToBase64()],
    )
    input_utterance = dataset[0].messages[0]["content"]
    assert input_utterance == "data:image/jpeg;base64,..."
