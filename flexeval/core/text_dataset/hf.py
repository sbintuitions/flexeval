from __future__ import annotations

from typing import Iterator

import datasets

from .base import TextDataset


class HfTextDataset(TextDataset):
    """
    This class represents a dataset of text examples loaded from Hugging Face datasets.

    Args:
        dataset_name: The name of the dataset to load.
        split: The split of the dataset to load.
        field: The field to extract from the dataset.
        subset: The subset of the dataset to load.
    """

    def __init__(self, dataset_name: str, split: str, field: str, subset: str | None = None) -> None:
        self._dataset = datasets.load_dataset(dataset_name, split=split, name=subset)[field]
        if not isinstance(self._dataset[0], str):
            msg = f"field '{field}' is not string type"
            raise TypeError(msg)

    def __iter__(self) -> Iterator[str]:
        yield from self._dataset
