from __future__ import annotations

from typing import Any

import pytest

from flexeval.core.reward_bench_dataset.template_based import (
    HFRewardBenchDataset,
    JsonlRewardBenchDataset,
    TemplateRewardBenchDataset,
)

DATASETS_TO_TEST = [
    (
        HFRewardBenchDataset,
        {
            "path": "tests/dummy_modules/hf_dataset",
            "split": "train",
        },
    ),
    (
        JsonlRewardBenchDataset,
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
    dataset_class: type[TemplateRewardBenchDataset],
    kwargs: dict[str, Any],
) -> None:
    dataset = dataset_class(
        **kwargs,
        prompt_template="{{ question }}",
        chosen_template="{{ answers[0] }}",
        rejected_template="{{ answers[-1] }}",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert item.prompt == [{"role": "user", "content": "What is the highest mountain in the world."}]
    assert item.chosen == [{"role": "assistant", "content": "Mount Everest"}]
    assert item.rejected == [{"role": "assistant", "content": "Everest"}]

    item = dataset[1]
    assert item.prompt == [{"role": "user", "content": "What is the chemical symbol for water?"}]
    assert item.chosen == [{"role": "assistant", "content": "H2O"}]
    assert item.rejected == [{"role": "assistant", "content": "H2O"}]

    item = dataset[4]
    assert item.prompt == [{"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}]
    assert item.chosen == [{"role": "assistant", "content": "William Shakespeare"}]
    assert item.rejected == [{"role": "assistant", "content": "Shakespeare"}]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_data_range(
    dataset_class: type[TemplateRewardBenchDataset],
    kwargs: dict[str, Any],
) -> None:
    data_range = (2, 5)
    dataset = dataset_class(
        **kwargs,
        prompt_template="{{ question }}",
        chosen_template="{{ answers[0] }}",
        rejected_template="{{ answers[-1] }}",
        data_range=data_range,
    )
    assert list(range(*data_range)) == [i.extra_info["id"] for i in dataset]


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_keep_conditions(dataset_class: type[TemplateRewardBenchDataset], kwargs: dict[str, Any]) -> None:
    original_dataset = dataset_class(
        **kwargs,
        prompt_template="{{ question }}",
        chosen_template="{{ answers[0] }}",
        rejected_template="{{ answers[-1] }}",
    )

    filtered_dataset = dataset_class(
        **kwargs,
        prompt_template="{{ question }}",
        chosen_template="{{ answers[0] }}",
        rejected_template="{{ answers[-1] }}",
        keep_conditions={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.extra_info["answers"]) == 1


@pytest.mark.parametrize(
    ("dataset_class", "kwargs"),
    DATASETS_TO_TEST,
)
def test_remove_conditions(dataset_class: type[TemplateRewardBenchDataset], kwargs: dict[str, Any]) -> None:
    original_dataset = dataset_class(
        **kwargs,
        prompt_template="{{ question }}",
        chosen_template="{{ answers[0] }}",
        rejected_template="{{ answers[-1] }}",
    )

    filtered_dataset = dataset_class(
        **kwargs,
        prompt_template="{{ question }}",
        chosen_template="{{ answers[0] }}",
        rejected_template="{{ answers[-1] }}",
        remove_conditions={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.extra_info["answers"]) > 1
