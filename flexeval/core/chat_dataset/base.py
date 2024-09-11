from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class ChatInstance:
    """
    A dataclass representing a single chat that will be fed to a chat language model.
    """

    messages: list[dict[str, str]]
    """
    A list of messages in the chat.
    The format of messages typically follows [OpenAI's Chat Completions API](https://platform.openai.com/docs/guides/text-generation/chat-completions-api).
    ```json
    [
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        {
            "role": "user",
            "content": "I'd like to book a flight to Paris."
        }
    ]
    ```
    """
    references: list[str]
    """
    A list of reference responses to the user's last message.
    The model's response will be evaluated against these references.
    """
    extra_info: dict[str, Any]
    """
    Extra information that can be used by passing to `Metric`.
    """

    def __post_init__(self) -> None:
        if "messages" in self.extra_info:
            msg = (
                "'extra_info' in ChatInstance cannot contain a key named 'messages', "
                "as it will conflict with the 'messages' attribute. "
                "The key 'messages' will be removed."
            )
            warnings.warn(msg, stacklevel=2)
            self.extra_info.pop("messages")

    @property
    def inputs(self) -> list[dict[str, str]]:
        """
        Alias for `messages`.
        This is used in `FewShotGenerator` so that it can access the inputs with the same attribute name as
        `GenerationInstance` and `MultipleChoiceInstance`.
        """
        return self.messages


class ChatDataset(Sequence[ChatInstance], ABC):
    """A dataset holding `ChatInstance`."""

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of chat instances in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int) -> ChatInstance:
        """
        Returns the i-th chat instance.
        """
        raise NotImplementedError

    @abstractmethod
    def require_incremental_response(self) -> bool:
        """If true, the inputs consist of multiple user utterances and the
        model should generate responses for each utterance incrementally.

        Otherwise, the model just has to continue the conversation from the last user utterance.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"
