from flexeval.core.multiple_choice_dataset.hf_dataset import HFMultipleChoiceDataset


def test_hf_multiple_choice_dataset() -> None:
    dataset = HFMultipleChoiceDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
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


def test_test_template_filters() -> None:
    original_dataset = HFMultipleChoiceDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
    )

    filtered_dataset = HFMultipleChoiceDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        choices_templates=["{{ answers[0] }}"],
        answer_index_template="0",
        template_filters={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.inputs["answers"]) == 1
