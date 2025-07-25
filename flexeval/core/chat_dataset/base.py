from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ChatInstance:
    """
    A dataclass representing a single chat that will be fed to a chat language model.
    """

    messages: list[dict[str, Any]]
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

    Tool-Calling message must follow the same format as the OpenAI ChatCompletion API.
    https://platform.openai.com/docs/guides/function-calling?api-mode=chat#defining-functions
    ```json
    {
        "role": "assistant",
        "content": "content", # `None` is also allowed if `tool_calls` exists.
        "tool_calls": [
            {
                "id": "dummy1",
                "function": {
                    "name": "search_web",
                    "arguments": "{\"query\": \"flexeval developer\"}"  # Note that this is a json string, not a dictionary.
                }
            }
        ]
    }
    ```

    The results from tools should be represented as messages with the role "tool":
    ```
    {
        "role": "tool",
        "tool_call_id": "dummy1", # Optional, models on OpenAI APIs requires this field.
        "name": "search_web", # Optional, Some HuggingFace models require this field.
        "content": "[{\"title\": \"sbintuitions/flexeval: Flexible evaluation tool...\", \"description\": \"...\"}]",
    }
    """  # noqa: E501
    tools: list[dict[str, Any]] | None = None
    """
    A list of definitions of tools in the chat.
    The format of tools typically follows [OpenAI's Chat Completion API](https://platform.openai.com/docs/guides/function-calling#function-calling-steps)
    Currently, only function calling (tools with type="function") is supported.
    """
    references: list[str] = field(default_factory=list)
    """
    A list of reference responses to the user's last message.
    The model's response will be evaluated against these references.
    """
    extra_info: dict[str, Any] = field(default_factory=dict)
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

    def require_incremental_response(self) -> bool:
        """If true, the inputs consist of multiple user utterances and the
        model should generate responses for each utterance incrementally.

        Otherwise, the model just has to continue the conversation from the last user utterance.
        """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"
