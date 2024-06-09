from __future__ import annotations

from flexeval.core.generation_dataset import SacreBleuDataset


def test_sacrebleu_dataset() -> None:
    dataset = SacreBleuDataset(
        dataset_name="wmt20",
        langpair="en-ja",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert "source" in item.inputs
