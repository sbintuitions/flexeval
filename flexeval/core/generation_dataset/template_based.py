from __future__ import annotations

import json
from ast import literal_eval
from typing import Any

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import GenerationDataset, GenerationInstance


class TemplateGenerationDataset(GenerationDataset):
    """
    Load GenerationInstances from a JSONL file.

    Args:
        items: A list of dict items.
        reference_template: Specify the Jinja2 template to render the reference string
            if the dataset has a single reference.
        reference_list_template: Specify the Jinja2 template to render a list of reference strings
            if the dataset has multiple references.
        input_templates: A dictionary of Jinja2 templates for the inputs.
        data_range: The range of data to use.
        keep_conditions: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
        remove_conditions: A dictionary to indicate the condition to remove certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to remove.
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        input_templates: dict[str, str] | None = None,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        if reference_template and reference_list_template:
            msg = "Only one of reference_template and reference_list_template can be set."
            raise ValueError(msg)

        if data_range:
            start, end = data_range
            items = items[start:end]

        keep_conditions = keep_conditions or {}
        for template_str, value_to_keep in keep_conditions.items():
            key_template = JINJA2_ENV.from_string(template_str)
            items = [item for item in items if key_template.render(**item) == value_to_keep]
        remove_conditions = remove_conditions or {}
        for template_str, value_to_remove in remove_conditions.items():
            key_template = JINJA2_ENV.from_string(template_str)
            items = [item for item in items if key_template.render(**item) != value_to_remove]

        self.items = items
        input_templates = input_templates or {}
        self.input_templates: dict[str, Template] = {k: JINJA2_ENV.from_string(v) for k, v in input_templates.items()}
        self.reference_template = JINJA2_ENV.from_string(reference_template) if reference_template else None
        self.reference_list_template = (
            JINJA2_ENV.from_string(reference_list_template) if reference_list_template else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> GenerationInstance:
        item = self.items[i]
        inputs = dict(item.items())
        inputs.update({k: v.render(**item) for k, v in self.input_templates.items()})

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


class HFGenerationDataset(TemplateGenerationDataset):
    """
    Load GenerationInstances from a huggingface dataset.

    Args:
        path: The path to the Hugging Face dataset.
        split: The split of the dataset.
        subset: The subset of the dataset.
        dataset_kwargs: The additional keyword arguments for loading the dataset.
    """

    def __init__(
        self,
        path: str,
        split: str,
        subset: str | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        input_templates: dict[str, str] | None = None,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        dataset_kwargs = dataset_kwargs or {}
        dataset = datasets.load_dataset(path, name=subset, split=split, **dataset_kwargs)
        items = [dict(item) for item in dataset]

        super().__init__(
            items=items,
            reference_template=reference_template,
            reference_list_template=reference_list_template,
            input_templates=input_templates,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )


class JsonlGenerationDataset(TemplateGenerationDataset):
    """
    Load GenerationInstances from a JSONL file.

    Args:
        path: The path to the JSONL file.
    """

    def __init__(
        self,
        path: str,
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        input_templates: dict[str, str] | None = None,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        with open(path) as f:
            items = [json.loads(line) for line in f]

        super().__init__(
            items=items,
            reference_template=reference_template,
            reference_list_template=reference_list_template,
            input_templates=input_templates,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )
