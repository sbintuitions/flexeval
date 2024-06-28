from __future__ import annotations

import functools
from ast import literal_eval
from typing import Any

import datasets
import jinja2
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
        template_filters: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
    """

    def __init__(
        self,
        path: str,
        split: str,
        references_template: str,
        input_templates: dict[str, str] | None = None,
        subset: str | None = None,
        template_filters: dict[str, str] | None = None,
    ) -> None:
        self._dataset = datasets.load_dataset(path, name=subset, split=split)

        template_filters = template_filters or {}
        for template_str, value_to_keep in template_filters.items():

            def _to_keep_this_item(item: dict[str, Any], filter_template: jinja2.Template, value_to_keep: str) -> bool:
                return filter_template.render(**item) == value_to_keep

            self._dataset = self._dataset.filter(
                functools.partial(
                    _to_keep_this_item,
                    filter_template=JINJA2_ENV.from_string(template_str),
                    value_to_keep=value_to_keep,
                ),
            )

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
