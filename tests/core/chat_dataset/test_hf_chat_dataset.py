from flexeval.core.chat_dataset import HFChatDataset


def test_hf_dataset() -> None:
    system_message = "You are a quiz player."
    chat_dataset = HFChatDataset(
        dataset_name="llm-book/aio",
        split="validation[:10]",
        input_template="{{question}}",
        references_template="{{ answers }}",
        extra_info_templates={"section": "{{ section }}"},
        system_message_template=system_message,
    )

    assert len(chat_dataset) == 10

    assert chat_dataset[0].messages == [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": "映画『ウエスト・サイド物語』に登場する2つの少年グループといえば、シャーク団と何団?",
        },
    ]
    assert chat_dataset[0].references == ["ジェット団"]
    assert chat_dataset[0].extra_info == {"section": "開発データ問題"}
