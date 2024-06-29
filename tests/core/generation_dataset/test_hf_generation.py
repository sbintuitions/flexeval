from flexeval.core.generation_dataset import HFGenerationDataset


def test_hf_dataset() -> None:
    dataset = HFGenerationDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        input_templates={"additional_input": "added_question: {{question}}"},
        references_template="{{ answers }}",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert item.inputs == {
        "question": "What is the highest mountain in the world.",
        "answers": ["Mount Everest", "Everest"],
        "additional_input": "added_question: What is the highest mountain in the world.",
    }
    assert item.references == ["Mount Everest", "Everest"]


def test_test_template_filters() -> None:
    original_dataset = HFGenerationDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        references_template="",
    )

    filtered_dataset = HFGenerationDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        references_template="",
        template_filters={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.inputs["answers"]) == 1
