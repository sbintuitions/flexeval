from __future__ import annotations

from typing import Iterator

from flexeval.core.text_dataset import TextDataset


class DummyTextDataset(TextDataset):
    def __iter__(self) -> Iterator[str]:
        yield "This is a test."
        yield "This is another test."
