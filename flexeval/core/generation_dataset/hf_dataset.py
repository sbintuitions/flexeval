from __future__ import annotations

from ast import literal_eval

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_utils import JINJA2_ENV, get_template_filter_function

from .base import GenerationDataset, GenerationInstance


class HFGenerationDataset(GenerationDataset):
    """
    A dataset for generation tasks using Hugging Face datasets.

    Args:
        path: The name or path of the huggingface dataset.
        split: The split of the dataset to use.
        reference_template: Specify the Jinja2 template to render the reference string
            if the dataset has a single reference.
        reference_list_template: Specify the Jinja2 template to render a list of reference strings
            if the dataset has multiple references.
        input_templates: A dictionary of Jinja2 templates for the inputs.
        subset: The subset of the dataset to use.
        template_filters: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
    """

    def __init__(
        self,
        path: str,
        split: str,
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        input_templates: dict[str, str] | None = None,
        subset: str | None = None,
        template_filters: dict[str, str] | None = None,
    ) -> None:
        if reference_template and reference_list_template:
            msg = "Only one of reference_template and reference_list_template can be set."
            raise ValueError(msg)

        self.dataset = datasets.load_dataset(path, name=subset, split=split)

        template_filters = template_filters or {}
        for template_str, value_to_keep in template_filters.items():
            self.dataset = self.dataset.filter(get_template_filter_function(template_str, value_to_keep))

        input_templates = input_templates or {}
        self.input_templates: dict[str, Template] = {k: JINJA2_ENV.from_string(v) for k, v in input_templates.items()}

        self.reference_template = JINJA2_ENV.from_string(reference_template) if reference_template else None
        self.reference_list_template = (
            JINJA2_ENV.from_string(reference_list_template) if reference_list_template else None
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> GenerationInstance:
        item = self.dataset[i]
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
