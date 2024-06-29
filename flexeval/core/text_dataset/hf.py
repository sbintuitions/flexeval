from __future__ import annotations

import datasets

from flexeval.core.utils.jinja2_utils import JINJA2_ENV, get_template_filter_function

from .base import TextDataset


class HFTextDataset(TextDataset):
    """
    This class represents a dataset of text examples loaded from Hugging Face datasets.

    Args:
        path: The name of the dataset to load.
        split: The split of the dataset to load.
        text_template: A Jinja2 template for the text.
        subset: The subset of the dataset to load.
        template_filters: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
    """

    def __init__(
        self,
        path: str,
        split: str,
        text_template: str,
        subset: str | None = None,
        template_filters: dict[str, str] | None = None,
    ) -> None:
        self._dataset = datasets.load_dataset(path, split=split, name=subset)

        template_filters = template_filters or {}
        for template_str, value_to_keep in template_filters.items():
            self._dataset = self._dataset.filter(get_template_filter_function(template_str, value_to_keep))

        self._text_template = JINJA2_ENV.from_string(text_template)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> str:
        item = self._dataset[i]
        return self._text_template.render(**item)
