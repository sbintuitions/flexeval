from __future__ import annotations

import json
from os import PathLike

from .base import TextDataset


class JsonlTextDataset(TextDataset):
    """
    This class represents a dataset of text examples loaded from a JSONL file.

    Args:
        path: The path to the JSONL file.
        field: The field to extract from the JSONL file.
    """

    def __init__(self, path: str | PathLike[str], field: str) -> None:
        self._text_list: list[str] = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                self._text_list.append(item[field])

    def __len__(self) -> int:
        return len(self._text_list)

    def __getitem__(self, item: int) -> str:
        return self._text_list[item]
