from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall(ABC):
    id: int | None = None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class FunctionToolCall(ToolCall):
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ParsedToolCallingMessage:
    validation_result: Literal[
        "CompleteToolCall",
        "InCompleteToolCall",
        "TextOnly",
    ]
    text: str = None
    raw_text: str = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tool_call_dicts = [tool_call.to_dict() for tool_call in self.tool_calls]


class ToolParser(ABC):
    """
    An interface class used to extract tool calls from the model's output.
    """

    @abstractmethod
    def __call__(self, text: str) -> ParsedToolCallingMessage:
        """
        Extract tool_calls from the input text.

        Args:
            text: The text to process.
        """
        raise NotImplementedError
