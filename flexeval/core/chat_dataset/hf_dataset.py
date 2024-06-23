from __future__ import annotations

from ast import literal_eval

import datasets
from jinja2 import Template

from flexeval.core.utils.jinja2_env import JINJA2_ENV

from .base import ChatDataset, ChatInstance


class HFChatDataset(ChatDataset):
    """
    A chat dataset using Hugging Face datasets.
    This class only supports single-turn chat.

    Args:
        path: The name or path of the huggingface dataset.
        split: The split of the dataset to use.
        input_template: A Jinja2 template for the user input.
        references_template: A Jinja2 template for the references.
        subset: The subset of the dataset to use.
        require_incremental_response: Whether the dataset requires incremental response.
    """

    def __init__(
        self,
        path: str,
        split: str,
        input_template: str,
        references_template: str | None = None,
        subset: str | None = None,
        require_incremental_response: bool = False,
        extra_info_templates: dict[str, str] | None = None,
        system_message_template: str | None = None,
    ) -> None:
        self._dataset = datasets.load_dataset(path, name=subset, split=split)

        self._input_template: Template = JINJA2_ENV.from_string(input_template)

        self._references_template: Template | None = (
            JINJA2_ENV.from_string(references_template) if references_template else None
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
        return len(self._dataset)

    def __getitem__(self, i: int) -> ChatInstance:
        item = self._dataset[i]
        input_utterance = self._input_template.render(**item)
        messages = [{"role": "user", "content": input_utterance}]

        if self._system_message_template:
            system_message = self._system_message_template.render(**item)
            messages.insert(0, {"role": "system", "content": system_message})

        references = []
        if self._references_template:
            reference_string = self._references_template.render(**item)
            if reference_string.startswith("[") and reference_string.endswith("]"):
                references = literal_eval(reference_string)
            else:
                references = [reference_string]

        extra_info = dict(item.items())
        extra_info_from_templates = {
            key: template.render(**item) for key, template in self._extra_info_templates.items()
        }
        extra_info.update(extra_info_from_templates)

        return ChatInstance(messages=messages, references=references, extra_info=extra_info)
