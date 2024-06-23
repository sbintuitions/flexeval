from __future__ import annotations

from flexeval.core.chat_dataset import SacreBleuChatDataset


def test_sacrebleu_dataset() -> None:
    dataset = SacreBleuChatDataset(
        name="wmt20",
        langpair="en-ja",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert isinstance(item.messages, list)
    assert isinstance(item.references, list)
    assert isinstance(item.references[0], str)
