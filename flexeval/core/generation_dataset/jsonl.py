from __future__ import annotations

import json
from ast import literal_eval

from flexeval.core.utils.jinja2_env import JINJA2_ENV

from .base import GenerationDataset, GenerationInstance


class JsonlGenerationDataset(GenerationDataset):
    """
    Load GenerationInstances from a JSONL file.

    Args:
        path: The path to the JSONL file.
        references_template: A Jinja2 template for the references.
        data_range: The range of data to use.
    """

    def __init__(
        self,
        path: str,
        references_template: str,
        data_range: tuple[int, int] | None = None,
    ) -> None:
        with open(path) as f:
            self._dataset = [json.loads(line) for line in f]

        if data_range:
            start, end = data_range
            self._dataset = self._dataset[start:end]

        self._references_template = JINJA2_ENV.from_string(references_template)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> GenerationInstance:
        item = self._dataset[i]
        inputs = dict(item.items())

        reference_string = self._references_template.render(**item)
        if reference_string.startswith("[") and reference_string.endswith("]"):
            references = literal_eval(reference_string)
        else:
            references = [reference_string]
        return GenerationInstance(inputs=inputs, references=references)
