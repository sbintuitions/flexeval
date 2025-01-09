from __future__ import annotations

import json
from typing import Any

import datasets
from jinja2 import Template
from smart_open import open

from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import RewardBenchDataset, RewardBenchInstance


class TemplateRewardBenchDataset(RewardBenchDataset):
    """
    A chat dataset using Hugging Face datasets.
    This class only supports single-turn chat.

    Args:
        items: A list of items in a dict format.
        prompt_template: A Jinja2 template for the prompt.
        chosen_template: A Jinja2 template for the chosen response.
        rejected_template: A Jinja2 template for the rejected response.
        extra_info_templates: A dictionary of Jinja2 templates for extra information.
        data_range: The range of data to use.
        keep_conditions: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
        remove_conditions: A dictionary to indicate the condition to remove certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to remove.
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        prompt_template: str,
        chosen_template: str,
        rejected_template: str,
        category_template: str | None = None,
        extra_info_templates: dict[str, str] | None = None,
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

        self.prompt_template = JINJA2_ENV.from_string(prompt_template)
        self.chosen_template = JINJA2_ENV.from_string(chosen_template)
        self.rejected_template = JINJA2_ENV.from_string(rejected_template)
        self.category_template = None
        if category_template:
            self.category_template = JINJA2_ENV.from_string(category_template)

        extra_info_templates = extra_info_templates or {}
        self._extra_info_templates: dict[str, Template] = {
            key: JINJA2_ENV.from_string(template) for key, template in extra_info_templates.items()
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> RewardBenchInstance:
        item = self.items[i]

        prompt = self.prompt_template.render(**item)
        chosen = self.chosen_template.render(**item)
        rejected = self.rejected_template.render(**item)

        extra_info = dict(item.items())
        extra_info_from_templates = {
            key: template.render(**item) for key, template in self._extra_info_templates.items()
        }
        extra_info.update(extra_info_from_templates)

        return RewardBenchInstance(
            prompt=[{"role": "user", "content": prompt}],
            chosen=[{"role": "assistant", "content": chosen}],
            rejected=[{"role": "assistant", "content": rejected}],
            category_key=self.category_template.render(**item) if self.category_template else None,
            extra_info=extra_info,
        )


class HFRewardBenchDataset(TemplateRewardBenchDataset):
    """
    Load RewardBenchInstances from a Hugging Face dataset.

    Args:
        path: The path to the Hugging Face dataset.
        split: The split of the dataset.
        subset: The subset of the dataset.
        dataset_kwargs: The keyword arguments to pass to the Hugging Face dataset.
    """

    def __init__(
        self,
        path: str,
        split: str,
        subset: str | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        prompt_template: str = "{{ prompt }}",
        chosen_template: str = "{{ chosen }}",
        rejected_template: str = "{{ rejected }}",
        category_template: str | None = None,
        extra_info_templates: dict[str, str] | None = None,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        dataset_kwargs = dataset_kwargs or {}
        dataset = datasets.load_dataset(path, name=subset, split=split, **dataset_kwargs)
        items = [dict(item) for item in dataset]

        super().__init__(
            items=items,
            prompt_template=prompt_template,
            chosen_template=chosen_template,
            rejected_template=rejected_template,
            category_template=category_template,
            extra_info_templates=extra_info_templates,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )


class JsonlRewardBenchDataset(TemplateRewardBenchDataset):
    """
    Load RewardBenchInstances from a JSONL file.

    Args:
        path: The path to the JSONL file.
    """

    def __init__(
        self,
        path: str,
        prompt_template: str = "{{ prompt }}",
        chosen_template: str = "{{ chosen }}",
        rejected_template: str = "{{ rejected }}",
        category_template: str | None = None,
        extra_info_templates: dict[str, str] | None = None,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        with open(path) as f:
            items = [json.loads(line) for line in f]

        super().__init__(
            items=items,
            prompt_template=prompt_template,
            chosen_template=chosen_template,
            rejected_template=rejected_template,
            category_template=category_template,
            extra_info_templates=extra_info_templates,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )
