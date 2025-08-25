from __future__ import annotations

from typing import Any

import pytest

from flexeval.core.chat_dataset import HFChatDataset, JsonlChatDataset, TemplateChatDataset

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
