from __future__ import annotations

from typing import Any

import pytest

from flexeval.core.chat_dataset import HFChatDataset, JsonlChatDataset, TemplateChatDataset

DATASETS_TO_TEST = [
    (
        HFChatDataset,
        {
            "path": "tests/dummy_modules/hf_dataset",
            "split": "train",
        },
    ),
    (
        JsonlChatDataset,
        {
            "path": "tests/dummy_modules/test.jsonl",
        },
    ),
]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_template_dataset_with_reference(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
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


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_hf_dataset_with_reference_list(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
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


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_test_keep_conditions(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
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
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_test_remove_conditions(
    dataset_class: type[TemplateChatDataset],
    kwargs: dict[str, Any],
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