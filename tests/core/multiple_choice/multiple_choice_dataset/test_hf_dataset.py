from flexeval.core.multiple_choice_dataset.hf_dataset import HFMultipleChoiceDataset


def test_hf_multiple_choice_dataset() -> None:
    dataset = HFMultipleChoiceDataset(
        dataset_name="llm-book/JGLUE",
        subset="JCommonsenseQA",
        split="validation",
        input_templates={"test_additional_input": "additional: {{ question }}"},
        choices_templates=["{{ choice0 }}", "{{ choice1 }}", "{{ choice2 }}", "{{ choice3 }}", "{{ choice4 }}"],
        answer_index_template="{{ label }}",
    )

    assert len(dataset) > 0
    item = dataset[0]
    assert item.inputs == {
        "choice0": "掲示板",
        "choice1": "パソコン",
        "choice2": "マザーボード",
        "choice3": "ハードディスク",
        "choice4": "まな板",
        "label": 2,
        "q_id": 8939,
        "question": "電子機器で使用される最も主要な電子回路基板の事をなんと言う？",
        "test_additional_input": "additional: 電子機器で使用される最も主要な電子回路基板の事をなんと言う？",
    }
    assert item.choices == ["掲示板", "パソコン", "マザーボード", "ハードディスク", "まな板"]
    assert item.answer_index == 2
