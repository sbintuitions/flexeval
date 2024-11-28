from __future__ import annotations

from typing import Any

import pytest

from flexeval.core.multiple_choice_dataset import (
    HFMultipleChoiceDataset,
    JsonlMultipleChoiceDataset,
    TemplateMultipleChoiceDataset,
)

DATASETS_TO_TEST = [
    (
        HFMultipleChoiceDataset,
        {
            "path": "tests/dummy_modules/hf_dataset",
            "split": "train",
        },
    ),
    (
        JsonlMultipleChoiceDataset,
        {
            "path": "tests/dummy_modules/test.jsonl",
        },
    ),
]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_template_multiple_choice_dataset(
    dataset_class: type[TemplateMultipleChoiceDataset],
    kwargs: dict[str, Any],
) -> None:
    dataset = dataset_class(
        **kwargs,
        input_templates={"test_additional_input": "additional: {{ question }}"},
        choices_templates=[
            "{% if answers | length > 0 %}{{ answers[0] }}{% endif %}",
            "{% if answers | length > 1 %}{{ answers[1] }}{% endif %}",
            "{% if answers | length > 2 %}{{ answers[2] }}{% endif %}",
        ],
        answer_index_template="0",
    )

    assert len(dataset) > 0
    item = dataset[0]
    assert item.inputs == {
        "id": 0,
        "question": "What is the highest mountain in the world.",
        "answers": ["Mount Everest", "Everest"],
        "test_additional_input": "additional: What is the highest mountain in the world.",
    }
    assert item.choices == ["Mount Everest", "Everest"]
    assert item.answer_index == 0

    item = dataset[1]
    assert item.inputs == {
        "id": 1,
        "question": "What is the chemical symbol for water?",
        "answers": ["H2O"],
        "test_additional_input": "additional: What is the chemical symbol for water?",
    }

    item = dataset[4]
    assert item.inputs == {
        "id": 4,
        "question": "Who wrote 'Romeo and Juliet'?",
        "answers": ["William Shakespeare", "Shakespeare"],
        "test_additional_input": "additional: Who wrote 'Romeo and Juliet'?",
    }

@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_data_range(
    dataset_class: type[TemplateMultipleChoiceDataset],
    kwargs: dict[str, Any],
) -> None:
    data_range = (2, 5)
    dataset = dataset_class(
        **kwargs,
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
        data_range=data_range,
    )
    assert list(range(*data_range)) == [i.inputs["id"] for i in dataset]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_keep_conditions(dataset_class: type[TemplateMultipleChoiceDataset], kwargs: dict[str, Any]) -> None:
    original_dataset = dataset_class(
        **kwargs,
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
    )

    filtered_dataset = dataset_class(
        **kwargs,
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
        keep_conditions={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.inputs["answers"]) == 1


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_remove_conditions(dataset_class: type[TemplateMultipleChoiceDataset], kwargs: dict[str, Any]) -> None:
    original_dataset = dataset_class(
        **kwargs,
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
    )

    filtered_dataset = dataset_class(
        **kwargs,
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
        remove_conditions={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.inputs["answers"]) > 1
