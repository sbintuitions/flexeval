from __future__ import annotations

from flexeval.core.chat_dataset import ChatDataset, ChatInstance


class DummyChatDataset(ChatDataset):
    def __init__(self, require_incremental_response: bool = False) -> None:
        self._data = [
            [{"role": "sysmtem", "text": "You are a helpful assistant."}, {"role": "user", "text": "Help me!"}],
            [{"role": "user", "text": "Hello, world!"}],
        ]
        self._require_incremental_response = require_incremental_response

    def require_incremental_response(self) -> bool:
        return self._require_incremental_response

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int) -> ChatInstance:
        return ChatInstance(self._data[item], references=["This is reference"], extra_info={})
