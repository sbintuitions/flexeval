from __future__ import annotations

from typing import Any

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_utils import JINJA2_ENV, get_template_filter_function

from .base import MultipleChoiceDataset, MultipleChoiceInstance


class HFMultipleChoiceDataset(MultipleChoiceDataset):
    """
    A dataset for multiple-choice tasks using Hugging Face datasets.

    Args:
        path: The name or path of the huggingface dataset.
        split: The split of the dataset to use.
        choices_templates: A list of Jinja2 templates for the choices.
        answer_index_template: A Jinja2 template for the index of the correct answer.
        input_templates: A dictionary of Jinja2 templates for the inputs.
        subset: The subset of the dataset to use.
        data_files: The data files to load.
        whitespace_before_choices: Whether to add a whitespace before each choice.
            Maybe necessary for language with whitespaces.
    """

    def __init__(
        self,
        path: str,
        split: str,
        choices_templates: list[str],
        answer_index_template: str,
        input_templates: dict[str, str] | None = None,
        subset: str | None = None,
        data_files: str | None = None,
        whitespace_before_choices: bool = False,
        template_filters: dict[str, str] | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        dataset_kwargs = dataset_kwargs or {}
        self._dataset = datasets.load_dataset(path, split=split, name=subset, data_files=data_files, **dataset_kwargs)

        template_filters = template_filters or {}
        for template_str, value_to_keep in template_filters.items():
            self._dataset = self._dataset.filter(get_template_filter_function(template_str, value_to_keep))

        input_templates = input_templates or {}
        self._input_templates: dict[str, Template] = {k: JINJA2_ENV.from_string(v) for k, v in input_templates.items()}
        self._choices_templates = [JINJA2_ENV.from_string(t) for t in choices_templates]
        self._answer_index_template = JINJA2_ENV.from_string(
            answer_index_template,
        )
        self._whitespace_before_choices = whitespace_before_choices

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> MultipleChoiceInstance:
        item = self._dataset[i]
        inputs = dict(item.items())
        inputs.update({k: v.render(**item) for k, v in self._input_templates.items()})

        choices = [t.render(**item) for t in self._choices_templates]
        if any(len(c) == 0 for c in choices):
            msg = f"choices must be non-empty, but got {choices}"
            raise ValueError(msg)
        if self._whitespace_before_choices:
            choices = [" " + c for c in choices]

        answer_index = int(self._answer_index_template.render(**item))

        return MultipleChoiceInstance(
            inputs=inputs,
            choices=choices,
            answer_index=answer_index,
        )
