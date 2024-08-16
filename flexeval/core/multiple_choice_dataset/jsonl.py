from __future__ import annotations

import json
from typing import Any

from jinja2 import Template

from flexeval.core.utils.jinja2_utils import JINJA2_ENV, get_template_filter_function

from .base import MultipleChoiceDataset, MultipleChoiceInstance


class JsonlMultipleChoiceDataset(MultipleChoiceDataset):
    """
    Load MultipleChoiceInstance from a JSONL file.

    Args:
        path: The path to the JSONL file.
        choices_templates: A list of Jinja2 templates for the choices.
        answer_index_template: A Jinja2 template for the index of the correct answer.
        input_templates: A dictionary of Jinja2 templates for the inputs.
        whitespace_before_choices: Whether to add a whitespace before each choice.
            Maybe necessary for language with whitespaces.
        template_filters: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
    """

    def __init__(
        self,
        path: str,
        choices_templates: list[str],
        answer_index_template: str,
        input_templates: dict[str, str] | None = None,
        whitespace_before_choices: bool = False,
        template_filters: dict[str, str] | None = None,
    ) -> None:
        self._dataset: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                self._dataset.append(json.loads(line))

        template_filters = template_filters or {}
        for template_str, value_to_keep in template_filters.items():
            filter_func = get_template_filter_function(template_str, value_to_keep)
            self._dataset = [item for item in self._dataset if filter_func(item)]

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
