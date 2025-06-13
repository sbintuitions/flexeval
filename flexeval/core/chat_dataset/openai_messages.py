from __future__ import annotations

import json
from typing import Any, Iterator

from .base import ChatDataset, ChatInstance


def read_lines_from_file(file_path: str) -> Iterator[str]:
    with open(file_path, encoding="utf-8") as fi:
        yield from fi


class OpenAIMessagesDataset(ChatDataset):
    """This class loads data with OpenAI-like format in jsonl file.
    The difference lies in that this class has 'tool_definition' field, in which
    available tools are listed.

    Parameters:
        file_path (str | list[str] | None): Path or list of paths to `.jsonl` file(s).
        message_key (str): Key used to extract the list of messages from each JSON object.
        tool_definitions_key (str | None): Key used to extract the list of tool definitions from each JSON object.
            Set to `None` (default) for data without tool_calls.
        drop_if_last_from_assistant (bool): If true, when the last utterance is given by assistant, drop it.

    In Jsonl, each line must have a following structure:

    {
      '<message_key>': [
        {
          'role': 'user',
          'content': 'こんにちわ。元気になる言葉を教えて下さい。'
        },
        {
          'role': 'assistant',
          'content': 'こんなのはどうでしょう。どんどんやってください！'
        }
      ],
      '<tool_definitions_key>': [
        {
          'type': 'function',
          'function': { ... }
        }
      ]
    }
    """

    def __init__(
        self,
        file_path: str | None = None,
        message_key: str = "messages",
        tool_definitions_key: str | None = None,
        drop_if_last_from_assistant: bool = False,
        require_incremental_response: bool = False
    ) -> None:
        self.conversations: list[ChatInstance] = []
        with open(file_path) as f:
            dataset = [json.loads(line) for line in f]
        for sample in dataset:
            tool_dicts = None
            if tool_definitions_key is not None:
                tool_dicts = sample.get(tool_definitions_key, None)

            messages: list[dict[str, Any]] = sample[message_key]
            if drop_if_last_from_assistant and messages[-1]["role"] == "assistant":
                messages = messages[:-1]
            self.conversations.append(ChatInstance(messages=messages, tools=tool_dicts))
        self.require_incremental_response = require_incremental_response

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> ChatInstance:
        return self.conversations[idx]

    def require_incremental_response(self) -> bool:
        return self.require_incremental_response
