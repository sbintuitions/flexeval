from __future__ import annotations

import json
from typing import Any

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import MultipleChoiceDataset, MultipleChoiceInstance


class TemplateMultipleChoiceDataset(MultipleChoiceDataset):
    """
    An abstract dataset class for multiple-choice tasks.
    This class generates multiple-choice instances from a dict item and Jinja2 templates.

    Args:
        items: A list of dict items.
        choices_templates: A list of Jinja2 templates for the choices.
        answer_index_template: A Jinja2 template for the index of the correct answer.
        input_templates: A dictionary of Jinja2 templates for the inputs.
        whitespace_before_choices: Whether to add a whitespace before each choice.
            Maybe necessary for language with whitespaces.
        data_range: The range of data to use.
        keep_conditions: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
        remove_conditions: A dictionary to indicate the condition to remove certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to remove.
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        choices_templates: list[str],
        answer_index_template: str,
        input_templates: dict[str, str] | None = None,
        whitespace_before_choices: bool = False,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
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
        self.choices_templates = [JINJA2_ENV.from_string(t) for t in choices_templates]
        self.answer_index_template = JINJA2_ENV.from_string(
            answer_index_template,
        )
        self.whitespace_before_choices = whitespace_before_choices

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> MultipleChoiceInstance:
        item = self.items[i]
        inputs = dict(item.items())
        inputs.update({k: v.render(**item) for k, v in self.input_templates.items()})

        choices = [t.render(**item) for t in self.choices_templates]
        if any(len(c) == 0 for c in choices):
            msg = f"choices must be non-empty, but got {choices}"
            raise ValueError(msg)
        if self.whitespace_before_choices:
            choices = [" " + c for c in choices]

        answer_index = int(self.answer_index_template.render(**item))

        return MultipleChoiceInstance(
            inputs=inputs,
            choices=choices,
            answer_index=answer_index,
        )


class HFMultipleChoiceDataset(TemplateMultipleChoiceDataset):
    """
    Load MultipleChoiceInstance from a huggingface dataset.

    Args:
        path: The name or path of the huggingface dataset.
        split: The split of the dataset to use.
        subset: The subset of the dataset to use.
        dataset_kwargs: The keyword arguments for loading the dataset.
    """

    def __init__(
        self,
        path: str,
        split: str,
        choices_templates: list[str],
        answer_index_template: str,
        input_templates: dict[str, str] | None = None,
        subset: str | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        whitespace_before_choices: bool = False,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        dataset_kwargs = dataset_kwargs or {}
        items = datasets.load_dataset(path, split=split, name=subset, **dataset_kwargs)
        items = [dict(item) for item in items]

        super().__init__(
            items=items,
            choices_templates=choices_templates,
            answer_index_template=answer_index_template,
            input_templates=input_templates,
            whitespace_before_choices=whitespace_before_choices,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )


class JsonlMultipleChoiceDataset(TemplateMultipleChoiceDataset):
    """
    Load MultipleChoiceInstance from a JSONL file.
    """

    def __init__(
        self,
        path: str,
        choices_templates: list[str],
        answer_index_template: str,
        input_templates: dict[str, str] | None = None,
        whitespace_before_choices: bool = False,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        with open(path) as f:
            items = [json.loads(line) for line in f]

        super().__init__(
            items=items,
            choices_templates=choices_templates,
            answer_index_template=answer_index_template,
            input_templates=input_templates,
            whitespace_before_choices=whitespace_before_choices,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )
