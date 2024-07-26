from __future__ import annotations

import json
from ast import literal_eval

from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import GenerationDataset, GenerationInstance


class JsonlGenerationDataset(GenerationDataset):
    """
    Load GenerationInstances from a JSONL file.

    Args:
        path: The path to the JSONL file.
        reference_template: Specify the Jinja2 template to render the reference string
            if the dataset has a single reference.
        reference_list_template: Specify the Jinja2 template to render a list of reference strings
            if the dataset has multiple references.
        data_range: The range of data to use.
    """

    def __init__(
        self,
        path: str,
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        data_range: tuple[int, int] | None = None,
    ) -> None:
        if reference_template and reference_list_template:
            msg = "Only one of reference_template and reference_list_template can be set."
            raise ValueError(msg)

        with open(path) as f:
            self._dataset = [json.loads(line) for line in f]

        if data_range:
            start, end = data_range
            self._dataset = self._dataset[start:end]

        self.reference_template = JINJA2_ENV.from_string(reference_template) if reference_template else None
        self.reference_list_template = (
            JINJA2_ENV.from_string(reference_list_template) if reference_list_template else None
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> GenerationInstance:
        item = self._dataset[i]
        inputs = dict(item.items())

        reference_list: list[str] = []
        if self.reference_template:
            reference_string = self.reference_template.render(**item)
            reference_list.append(reference_string)
        if self.reference_list_template:
            reference_list_string = self.reference_list_template.render(**item)
            if not (reference_list_string.startswith("[") and reference_list_string.endswith("]")):
                msg = (
                    f"The reference_list_template should render a list of strings "
                    f"but we got `{reference_list_string}`."
                )
                raise ValueError(msg)
            reference_list.extend([str(ref) for ref in literal_eval(reference_list_string)])
        return GenerationInstance(inputs=inputs, references=reference_list)
