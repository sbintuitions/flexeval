from __future__ import annotations

import json
from os import PathLike
from typing import Iterator

from .base import TextDataset


class JsonlTextDataset(TextDataset):
    """
    This class represents a dataset of text examples loaded from a JSONL file.

    Args:
        file_path: The path to the JSONL file.
        field: The field to extract from the JSONL file.
    """

    def __init__(self, file_path: str | PathLike[str], field: str) -> None:
        self._file_path = file_path
        self._field = field

    def __iter__(self) -> Iterator[str]:
        with open(self._file_path) as f:
            for line in f:
                item = json.loads(line)
                yield item[self._field]
