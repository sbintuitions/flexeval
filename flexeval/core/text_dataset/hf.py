from __future__ import annotations

from typing import Any

import datasets

from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import TextDataset


class HFTextDataset(TextDataset):
    """
    This class represents a dataset of text examples loaded from Hugging Face datasets.

    Args:
        path: The name of the dataset to load.
        split: The split of the dataset to load.
        text_template: A Jinja2 template for the text.
        subset: The subset of the dataset to load.
        keep_conditions: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
        remove_conditions: A dictionary to indicate the condition to remove certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to remove.
        dataset_kwargs: Additional keyword arguments for `datasets.load_dataset`.
    """

    def __init__(
        self,
        path: str,
        split: str,
        text_template: str,
        subset: str | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        dataset_kwargs = dataset_kwargs or {}
        self.dataset = datasets.load_dataset(path, split=split, name=subset, **dataset_kwargs)

        keep_conditions = keep_conditions or {}
        for template_str, value_to_keep in keep_conditions.items():
            filter_template = JINJA2_ENV.from_string(template_str)
            self.dataset = self.dataset.filter(lambda x, t=filter_template, v=value_to_keep: t.render(**x) == v)
        remove_conditions = remove_conditions or {}
        for template_str, value_to_remove in remove_conditions.items():
            filter_template = JINJA2_ENV.from_string(template_str)
            self.dataset = self.dataset.filter(lambda x, t=filter_template, v=value_to_remove: t.render(**x) != v)

        self.text_template = JINJA2_ENV.from_string(text_template)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> str:
        item = self.dataset[i]
        return self.text_template.render(**item)
