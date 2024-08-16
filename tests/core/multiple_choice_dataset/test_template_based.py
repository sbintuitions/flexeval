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
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
    )

    assert len(dataset) > 0
    item = dataset[0]
    assert item.inputs == {
        "question": "What is the highest mountain in the world.",
        "answers": ["Mount Everest", "Everest"],
        "test_additional_input": "additional: What is the highest mountain in the world.",
    }
    assert item.choices == ["Mount Everest"]
    assert item.answer_index == 0


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_test_template_filters(dataset_class: type[TemplateMultipleChoiceDataset], kwargs: dict[str, Any]) -> None:
    original_dataset = dataset_class(
        **kwargs,
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
    )

    filtered_dataset = dataset_class(
        **kwargs,
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
        template_filters={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.inputs["answers"]) == 1
