from __future__ import annotations

from typing import Any

import pytest

from flexeval.core.generation_dataset import HFGenerationDataset, JsonlGenerationDataset, TemplateGenerationDataset

DATASETS_TO_TEST = [
    (
        HFGenerationDataset,
        {
            "path": "tests/dummy_modules/hf_dataset",
            "split": "train",
        },
    ),
    (
        JsonlGenerationDataset,
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
    dataset_class: type[TemplateGenerationDataset],
    kwargs: dict[str, Any],
) -> None:
    dataset = dataset_class(
        **kwargs,
        input_templates={"additional_input": "added_question: {{question}}"},
        reference_template="{{ answers[0] }}",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert item.inputs == {
        "question": "What is the highest mountain in the world.",
        "answers": ["Mount Everest", "Everest"],
        "additional_input": "added_question: What is the highest mountain in the world.",
    }
    assert item.references == ["Mount Everest"]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_template_dataset_with_reference_list(
    dataset_class: type[TemplateGenerationDataset],
    kwargs: dict[str, Any],
) -> None:
    dataset = dataset_class(
        **kwargs,
        input_templates={"additional_input": "added_question: {{question}}"},
        reference_list_template="{{ answers }}",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert item.inputs == {
        "question": "What is the highest mountain in the world.",
        "answers": ["Mount Everest", "Everest"],
        "additional_input": "added_question: What is the highest mountain in the world.",
    }
    assert item.references == ["Mount Everest", "Everest"]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_test_template_filters(
    dataset_class: type[TemplateGenerationDataset],
    kwargs: dict[str, Any],
) -> None:
    original_dataset = dataset_class(
        **kwargs,
    )

    filtered_dataset = dataset_class(
        **kwargs,
        template_filters={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.inputs["answers"]) == 1
