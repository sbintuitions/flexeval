from __future__ import annotations

from flexeval.core.chat_dataset import ChatDataset, ChatInstance


class DummyChatDataset(ChatDataset):
    def __init__(self, require_incremental_response: bool = False, use_tools: bool = False) -> None:
        self._data = [
            [{"role": "sysmtem", "content": "You are a helpful assistant."}, {"role": "user", "content": "Help me!"}],
            [{"role": "user", "content": "Hello, world!"}],
            [{"role": "user", "content": "I'd like to book a flight to Paris."}],
        ]
        if use_tools:
            self._tools = [
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "description": "Search the Web for specified query.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string", "description": "str - query for search"}},
                                "required": ["query"],
                            },
                            "return": {"type": "string", "description": "snippets: list"},
                        },
                    }
                ]
                for _ in range(3)
            ]
        else:
            self._tools = [None] * 3
        self._require_incremental_response = require_incremental_response

    def require_incremental_response(self) -> bool:
        return self._require_incremental_response

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int) -> ChatInstance:
        return ChatInstance(self._data[item], tools=self._tools[item], references=["This is reference"], extra_info={})
