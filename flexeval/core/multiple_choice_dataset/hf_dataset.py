from __future__ import annotations

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_env import JINJA2_ENV

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
    ) -> None:
        self._dataset = datasets.load_dataset(
            path,
            split=split,
            name=subset,
            data_files=data_files,
        )

        # workaround for the column names with whitespaces
        # cf. https://huggingface.co/datasets/nlp-waseda/JMMLU/discussions/3
        for column_name in self._dataset.column_names:
            fixed_column_name = column_name.strip()
            if fixed_column_name != column_name:
                self._dataset = self._dataset.rename_column(
                    column_name,
                    fixed_column_name,
                )

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
        if not isinstance(choices, list):
            msg = f"choices must be list, but got {type(choices)}"
            raise TypeError(msg)
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
