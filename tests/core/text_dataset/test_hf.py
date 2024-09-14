from flexeval.core.text_dataset import HFTextDataset


def test_hf_text_dataset() -> None:
    dataset = HFTextDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        text_template="{{ question }}",
    )

    texts = list(dataset)
    assert len(texts) == 10
    assert isinstance(texts[0], str)


def test_keep_conditions() -> None:
    original_dataset = HFTextDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        text_template="{{ question }}",
    )

    filtered_dataset = HFTextDataset(
        path="tests/dummy_modules/hf_dataset",
        split="train",
        text_template="{{ question }}",
        keep_conditions={
            "{{ answers | length }}": "1",
        },
    )

    assert 0 < len(filtered_dataset) < len(original_dataset)
