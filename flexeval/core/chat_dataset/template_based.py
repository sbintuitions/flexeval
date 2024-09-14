from __future__ import annotations

import json
from ast import literal_eval
from typing import Any

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import ChatDataset, ChatInstance


class TemplateChatDataset(ChatDataset):
    """
    A chat dataset using Hugging Face datasets.
    This class only supports single-turn chat.

    Args:
        items: A list of items in a dict format.
        input_template: A Jinja2 template for the user input.
        reference_template: Specify the Jinja2 template to render the reference string
            if the dataset has a single reference.
        reference_list_template: Specify the Jinja2 template to render a list of reference strings
            if the dataset has multiple references.
        require_incremental_response: Whether the dataset requires incremental response.
        extra_info_templates: A dictionary of Jinja2 templates for extra information.
        system_message_template: A Jinja2 template for the system message.
        data_range: The range of data to use.
        keep_conditions: A dictionary to indicate the condition to filter certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to keep.
        remove_conditions: A dictionary to indicate the condition to remove certain items.
            The key is a Jinja2 template string to embed the item into a string, and the value is the value to remove.
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        input_template: str,
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        require_incremental_response: bool = False,
        extra_info_templates: dict[str, str] | None = None,
        system_message_template: str | None = None,
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

        self.input_template = JINJA2_ENV.from_string(input_template)
        self.reference_template = JINJA2_ENV.from_string(reference_template) if reference_template else None
        self.reference_list_template = (
            JINJA2_ENV.from_string(reference_list_template) if reference_list_template else None
        )

        extra_info_templates = extra_info_templates or {}
        self._extra_info_templates: dict[str, Template] = {
            key: JINJA2_ENV.from_string(template) for key, template in extra_info_templates.items()
        }

        self._system_message_template: Template | None = (
            JINJA2_ENV.from_string(system_message_template) if system_message_template else None
        )

        self._require_incremental_response = require_incremental_response

    def require_incremental_response(self) -> bool:
        return self._require_incremental_response

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> ChatInstance:
        item = self.items[i]
        input_utterance = self.input_template.render(**item)
        messages = [{"role": "user", "content": input_utterance}]

        if self._system_message_template:
            system_message = self._system_message_template.render(**item)
            messages.insert(0, {"role": "system", "content": system_message})

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

        extra_info = dict(item.items())
        extra_info_from_templates = {
            key: template.render(**item) for key, template in self._extra_info_templates.items()
        }
        extra_info.update(extra_info_from_templates)

        return ChatInstance(messages=messages, references=reference_list, extra_info=extra_info)


class HFChatDataset(TemplateChatDataset):
    """
    Load ChatInstances from a Hugging Face dataset.

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
        input_template: str,
        subset: str | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        require_incremental_response: bool = False,
        extra_info_templates: dict[str, str] | None = None,
        system_message_template: str | None = None,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        dataset_kwargs = dataset_kwargs or {}
        dataset = datasets.load_dataset(path, name=subset, split=split, **dataset_kwargs)
        items = [dict(item) for item in dataset]

        super().__init__(
            items=items,
            input_template=input_template,
            reference_template=reference_template,
            reference_list_template=reference_list_template,
            require_incremental_response=require_incremental_response,
            extra_info_templates=extra_info_templates,
            system_message_template=system_message_template,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )


class JsonlChatDataset(TemplateChatDataset):
    """
    Load ChatInstances from a JSONL file.

    Args:
        path: The path to the JSONL file.
    """

    def __init__(
        self,
        path: str,
        input_template: str,
        reference_template: str | None = None,
        reference_list_template: str | None = None,
        require_incremental_response: bool = False,
        extra_info_templates: dict[str, str] | None = None,
        system_message_template: str | None = None,
        data_range: tuple[int, int] | None = None,
        keep_conditions: dict[str, str] | None = None,
        remove_conditions: dict[str, str] | None = None,
    ) -> None:
        with open(path) as f:
            items = [json.loads(line) for line in f]

        super().__init__(
            items=items,
            input_template=input_template,
            reference_template=reference_template,
            reference_list_template=reference_list_template,
            require_incremental_response=require_incremental_response,
            extra_info_templates=extra_info_templates,
            system_message_template=system_message_template,
            data_range=data_range,
            keep_conditions=keep_conditions,
            remove_conditions=remove_conditions,
        )
