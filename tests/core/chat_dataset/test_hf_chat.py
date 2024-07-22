from flexeval.core.chat_dataset import HFChatDataset


def test_hf_dataset_with_reference() -> None:
    system_message = "You are a quiz player."
    chat_dataset = HFChatDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
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


def test_hf_dataset_with_reference_list() -> None:
    system_message = "You are a quiz player."
    chat_dataset = HFChatDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
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


def test_test_template_filters() -> None:
    original_dataset = HFChatDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
    )

    filtered_dataset = HFChatDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        input_template="{{question}}",
        reference_list_template="{{ answers }}",
        template_filters={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
    for item in filtered_dataset:
        assert len(item.references) == 1
