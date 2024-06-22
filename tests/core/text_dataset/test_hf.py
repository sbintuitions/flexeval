from flexeval.core.text_dataset import HFTextDataset


def test_hf_text_dataset() -> None:
    dataset = HFTextDataset(
        dataset_name="llm-book/aio",
        split="validation[:10]",
        field="question",
    )

    texts = list(dataset)
    assert len(texts) == 10
    assert isinstance(texts[0], str)
