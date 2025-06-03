from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


# The format of `tool_call_id` follows the OpenAI Chat-Completion API
def generate_tool_call_id() -> str:
    chars = string.ascii_letters + string.digits
    return f"call_{''.join(random.choices(chars, k=24))}"


@dataclass
class ToolCall(ABC):
    """
    An abstract base class representing a generic tool call.
    Subclasses should implement the `to_dict` method to return a dictionary representation of the tool call.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the tool call.

        Returns:
            A dictionary describing the tool call.
        """
        raise NotImplementedError


@dataclass
class FunctionToolCall(ToolCall):
    """
    Represents a function tool call with its name, arguments, and optional ID.

    Attributes:
        name: The name of the function to call.
        arguments: Arguments to pass to the function.
        id: An optional identifier for the tool call.
    """

    name: str
    arguments: dict[str, Any]
    id: int | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = generate_tool_call_id()

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the function tool call.

        Returns:
            A dictionary with the structure required for function tool calls.
        """
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ToolCallingMessage:
    """
    Represents the parsed result of a model output that may contain tool calls.

    Attributes:
        validation_result: The validation result of the parsing
            (e.g., 'CompleteToolCall', 'InCompleteToolCall', or 'TextOnly').
        text: The text remaining after extracting the tool-calling part.
        raw_text: The raw, unprocessed text.
        tool_calls: A list of ToolCall objects extracted from the text.
    """

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
    def __call__(self, text: str) -> ToolCallingMessage:
        """
        Extract tool_calls from the input text.

        Args:
            text: The text to process.
        """
        raise NotImplementedError
