from flexeval.core.text_dataset import HfTextDataset


def test_hf_text_dataset() -> None:
    dataset = HfTextDataset(
        dataset_name="llm-book/aio",
        split="validation[:10]",
        field="question",
    )

    texts = list(dataset)
    assert len(texts) == 10
    assert isinstance(texts[0], str)
