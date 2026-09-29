from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import datasets

from .base import ChatDataset, ChatInstance


def read_lines_from_file(file_path: str) -> Iterator[str]:
    with open(file_path, encoding="utf-8") as fi:
        yield from fi


def _normalize_hf_messages(messages: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Normalize a messages column loaded from a Hugging Face dataset into a list of message dicts.

    A list of structs is materialized as a dict of lists
    (e.g. `{"role": ["user", "assistant"], "content": ["hi", "hello"]}`)
    when the dataset was stored with a `Sequence` of a dict feature (Arrow `struct<list>`),
    regardless of the `datasets` version used to read it.
    Struct columns also fill missing optional fields (e.g. `tool_calls`) with `None`,
    so such fields are dropped to keep the messages compatible with the OpenAI format.
    """
    if isinstance(messages, dict):
        keys = list(messages.keys())
        messages = [dict(zip(keys, values)) for values in zip(*(messages[key] for key in keys))]
    if not isinstance(messages, list):
        msg = f"Messages must be a list of dicts or a dict of lists, but got {type(messages)}."
        raise TypeError(msg)
    return [{key: value for key, value in message.items() if value is not None} for message in messages]


def _load_samples(
    file_path: str | None,
    path: str | None,
    split: str | None,
    subset: str | None,
    dataset_kwargs: dict[str, Any] | None,
    message_key: str,
) -> list[dict[str, Any]]:
    """Load samples either from a jsonl file or from a Hugging Face dataset."""
    if (file_path is None) == (path is None):
        msg = "Exactly one of `file_path` or `path` must be specified."
        raise ValueError(msg)

    if file_path is not None:
        with open(file_path) as f:
            return [json.loads(line) for line in f]

    if split is None:
        msg = "`split` must be specified when loading from a Hugging Face dataset."
        raise ValueError(msg)
    dataset_kwargs = dataset_kwargs or {}
    hf_dataset = datasets.load_dataset(path, name=subset, split=split, **dataset_kwargs)
    samples = [dict(item) for item in hf_dataset]
    for sample in samples:
        sample[message_key] = _normalize_hf_messages(sample[message_key])
    return samples


class OpenAIMessagesDataset(ChatDataset):
    """This class loads data with OpenAI-like format from a jsonl file or a Hugging Face dataset.
    The difference lies in that this class has 'tool_definition' field, in which
    available tools are listed.

    Tool-Calling (Function-Calling) is supported in this class.
    It must follow the same format as the OpenAI ChatCompletion API.
    https://platform.openai.com/docs/guides/function-calling?api-mode=chat#defining-functions

    Parameters:
        file_path (str | None): Path to a `.jsonl` file. Mutually exclusive with `path`.
        message_key (str): Key used to extract the list of messages from each JSON object.
        tool_definitions_key (str | None): Key used to extract the list of tool definitions from each JSON object.
            Set to `None` (default) for data without tool_calls.
        drop_if_last_from_assistant (bool): If true, when the last utterance is given by assistant, drop it.
            And the last assistant utterance will be used as reference answer if `references_key` is not given.
        references_key (str | None): Key used to extract the reference answers from each JSON object.
        path (str | None): The path to a Hugging Face dataset. Mutually exclusive with `file_path`.
        split (str | None): The split of the Hugging Face dataset. Required when `path` is given.
        subset (str | None): The subset (config name) of the Hugging Face dataset.
        dataset_kwargs (dict[str, Any] | None): The keyword arguments to pass to `datasets.load_dataset`.

    All the fields other than `message_key`, `tool_definitions_key` and `references_key`
    are stored in `extra_info` of each `ChatInstance`.

    In Jsonl, each line must have a following structure:
    ```json
    {
      '<message_key>': [
        {
          'role': 'user',
          'content': 'こんにちは。元気が出る言葉を教えて下さい。'
        },
        {
          'role': 'assistant',
          'content': 'こんなのはどうでしょう。どんどんやってください！'
        },
      ],
    }
    ```

    Example with tool-calling:
    ```json
    {
      '<message_key>': [
        {
          'role': 'user',
          'content': 'こんにちは。元気が出る偉人の言葉を教えて下さい。'
        },
        {
          'role': 'assistant',
          'content': '調べてみますね。',
          'tool_calls': [
            {
              'id': 'dummy1',
              'function': {
                'name': 'web_search',
                'arguments': '{"query": "元気が出る言葉 偉人"}',
              }
            }
          ]
        }
      ],
      '<tool_definitions_key>': [
        {
          "type": "function",
          "function": {
            "name": "web_search",
            ...
          }
        }
      ]
    }
    ```

    Example with reference answers:
    ```json
    {
      '<message_key>': [
        {
          'role': 'user',
          'content': 'こんにちは。元気が出る言葉を教えて下さい。'
        },
      ],
      '<references_key>': [
        'こんなのはどうでしょう。どんどんやってください！',
        'こんなのはどうでしょう。頑張ってください！',
      ],
    }
    ```

    If there is only one reference answer for each conversation,
    it can also be directly given as a string instead of a list:
    ```json
    {
      '<message_key>': [
        {
          'role': 'user',
          'content': 'こんにちは。元気が出る言葉を教えて下さい。'
        },
      ],
      '<references_key>': 'こんなのはどうでしょう。どんどんやってください！',
    }
    ```

    Example of loading from a Hugging Face dataset with the same structure:
    ```python
    OpenAIMessagesDataset(
        path="ScaleAI/MultiChallenge",
        split="test",
        message_key="conversation",
    )
    ```
    """

    def __init__(
        self,
        file_path: str | None = None,
        message_key: str = "messages",
        tool_definitions_key: str | None = None,
        drop_if_last_from_assistant: bool = False,
        references_key: str | None = None,
        path: str | None = None,
        split: str | None = None,
        subset: str | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        dataset = _load_samples(
            file_path=file_path,
            path=path,
            split=split,
            subset=subset,
            dataset_kwargs=dataset_kwargs,
            message_key=message_key,
        )

        self.conversations: list[ChatInstance] = []
        for sample in dataset:
            tool_dicts = None
            if tool_definitions_key is not None:
                tool_dicts = sample.get(tool_definitions_key, None)

            messages: list[dict[str, Any]] = sample.pop(message_key)
            last_assistant_content: str | None = None
            if drop_if_last_from_assistant and messages[-1]["role"] == "assistant":
                last_assistant_content = messages[-1].get("content", None)
                messages = messages[:-1]

            if references_key:
                references = sample.pop(references_key, None)
                if isinstance(references, str):
                    references = [references]
                elif isinstance(references, list) and all(isinstance(ref, str) for ref in references):
                    pass
                else:
                    msg = "Invalid format for references."
                    raise ValueError(msg)
            elif references_key is None and last_assistant_content:
                references = [last_assistant_content]
            else:
                references = []

            self.conversations.append(
                ChatInstance(messages=messages, tools=tool_dicts, references=references, extra_info=sample)
            )

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> ChatInstance:
        return self.conversations[idx]
