from flexeval.core.generation_dataset import HFGenerationDataset


def test_hf_dataset() -> None:
    dataset = HFGenerationDataset(
        path="llm-book/aio",
        split="validation",
        input_templates={"additional_input": "追加の問題：{{question}}"},
        references_template="{{ answers }}",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert item.inputs == {
        "answers": ["ジェット団"],
        "competition": "第2回AI王",
        "number": 1,
        "original_additional_info": "",
        "original_answer": "ジェット団",
        "original_question": "映画『ウエスト・サイド物語』に登場する2つの少年グループといえば、シャーク団と何団？",
        "qid": "AIO02-0001",
        "question": "映画『ウエスト・サイド物語』に登場する2つの少年グループといえば、シャーク団と何団?",
        "section": "開発データ問題",
        "timestamp": "2021/01/29",
        "additional_input": "追加の問題：映画『ウエスト・サイド物語』に登場する2つの少年グループといえば、シャーク団と何団?",  # noqa: E501
    }

    assert item.references == ["ジェット団"]
