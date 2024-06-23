from __future__ import annotations

from ast import literal_eval

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_env import JINJA2_ENV

from .base import GenerationDataset, GenerationInstance


class HFGenerationDataset(GenerationDataset):
    """
    A dataset for generation tasks using Hugging Face datasets.

    Args:
        path: The name or path of the huggingface dataset.
        split: The split of the dataset to use.
        references_template: A Jinja2 template for the references.
        input_templates: A dictionary of Jinja2 templates for the inputs.
        subset: The subset of the dataset to use.
        max_lengths: If provided, filter out instances with lengths exceeding the specified values.
    """

    def __init__(
        self,
        path: str,
        split: str,
        references_template: str,
        input_templates: dict[str, str] | None = None,
        subset: str | None = None,
        max_lengths: dict[str, int] | None = None,
    ) -> None:
        self._dataset = datasets.load_dataset(path, name=subset, split=split)

        max_lengths = max_lengths or {}
        for key, max_length in max_lengths.items():
            self._dataset = self._dataset.filter(lambda item, k=key, max_l=max_length: len(item[k]) <= max_l)

        input_templates = input_templates or {}
        self._input_templates: dict[str, Template] = {k: JINJA2_ENV.from_string(v) for k, v in input_templates.items()}
        self._references_template = JINJA2_ENV.from_string(references_template)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> GenerationInstance:
        item = self._dataset[i]
        inputs = dict(item.items())
        inputs.update({k: v.render(**item) for k, v in self._input_templates.items()})

        reference_string = self._references_template.render(**item)
        if reference_string.startswith("[") and reference_string.endswith("]"):
            references = literal_eval(reference_string)
        else:
            references = [reference_string]
        return GenerationInstance(inputs=inputs, references=references)
