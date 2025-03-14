from __future__ import annotations

from flexeval.core.text_dataset import TextDataset, TextInstance


class DummyTextDataset(TextDataset):
    def __init__(self) -> None:
        self._text_list = [
            "This is a test.",
            "This is another test.",
        ]

    def __len__(self) -> int:
        return len(self._text_list)

    def __getitem__(self, item: int) -> TextInstance:
        return TextInstance(self._text_list[item])
