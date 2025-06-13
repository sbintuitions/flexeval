from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from typing import Any, Callable

import pytest

from flexeval.core.chat_dataset import ChatInstance, OpenAIMessagesDataset

TEST_CHAT_MESSAGES = [
    [
        {"role": "user", "content": "How can I find the best 401k plan for my needs?"},
        {"role": "assistant", "content": "The first step is to research your options. You should look at ..."},
        {"role": "user", "content": "Thank you for your help!"},
        {"role": "assistant", "content": "You're welcome!"},
    ]
]


@pytest.fixture()
def jsonl_data_factory(tmp_path) -> Callable:  # noqa: ANN001
    def _create(message_key: str, messages_list: list[dict], num_samples: int = 10) -> str:
        file_path = tmp_path / f"mock_data_{message_key}.jsonl"
        with open(file_path, "w") as f:
            for messages in messages_list * num_samples:
                f.write(json.dumps({message_key: messages}) + "\n")
        return str(file_path)

    return _create


@pytest.mark.parametrize("message_key", ["conversations", "messages", "chat", "dialog"])
def test_load_dataset_with_messages_key(jsonl_data_factory, message_key: str) -> None:  # noqa: ANN001
    tmp_jsonl_path = jsonl_data_factory(message_key, TEST_CHAT_MESSAGES)

    dataset = OpenAIMessagesDataset(file_path=tmp_jsonl_path, message_key=message_key)

    assert len(dataset) == 10

    assert dataset[0] == ChatInstance(
        messages=[
            {"role": TEST_CHAT_MESSAGES[0][0]["role"], "content": TEST_CHAT_MESSAGES[0][0]["content"]},
            {"role": TEST_CHAT_MESSAGES[0][1]["role"], "content": TEST_CHAT_MESSAGES[0][1]["content"]},
            {"role": TEST_CHAT_MESSAGES[0][2]["role"], "content": TEST_CHAT_MESSAGES[0][2]["content"]},
            {"role": TEST_CHAT_MESSAGES[0][3]["role"], "content": TEST_CHAT_MESSAGES[0][3]["content"]},
        ]
    )


def test_load_dataset_with_drop_if_last_from_assistant(jsonl_data_factory) -> None:  # noqa: ANN001
    tmp_jsonl_path = jsonl_data_factory("messages", TEST_CHAT_MESSAGES)

    dataset = OpenAIMessagesDataset(file_path=tmp_jsonl_path, message_key="messages", drop_if_last_from_assistant=True)

    assert len(dataset) == 10

    # If last message is from an assistant, drop it.
    assert dataset[0] == ChatInstance(
        messages=[
            {"role": TEST_CHAT_MESSAGES[0][0]["role"], "content": TEST_CHAT_MESSAGES[0][0]["content"]},
            {"role": TEST_CHAT_MESSAGES[0][1]["role"], "content": TEST_CHAT_MESSAGES[0][1]["content"]},
            {"role": TEST_CHAT_MESSAGES[0][2]["role"], "content": TEST_CHAT_MESSAGES[0][2]["content"]},
        ]
    )

    test_chat_messages_with_last_user = deepcopy(TEST_CHAT_MESSAGES)
    test_chat_messages_with_last_user[0].pop(-1)
    tmp_jsonl_path_with_last_user = jsonl_data_factory("messages", test_chat_messages_with_last_user)
    dataset_with_last_user = OpenAIMessagesDataset(
        file_path=tmp_jsonl_path_with_last_user, message_key="messages", drop_if_last_from_assistant=True
    )
    # The last utterance is kept intact if not from an assistant.
    assert dataset_with_last_user[0] == ChatInstance(
        messages=[
            {"role": TEST_CHAT_MESSAGES[0][0]["role"], "content": TEST_CHAT_MESSAGES[0][0]["content"]},
            {"role": TEST_CHAT_MESSAGES[0][1]["role"], "content": TEST_CHAT_MESSAGES[0][1]["content"]},
            {"role": TEST_CHAT_MESSAGES[0][2]["role"], "content": TEST_CHAT_MESSAGES[0][2]["content"]},
        ]
    )


def test_load_dataset_with_require_incremental_response(jsonl_data_factory) -> None:  # noqa: ANN001
    tmp_jsonl_path = jsonl_data_factory("messages")

    dataset = OpenAIMessagesDataset(file_path=tmp_jsonl_path)
    assert dataset.require_incremental_response is False

    dataset = OpenAIMessagesDataset(file_path=tmp_jsonl_path, require_incremental_response=True)
    assert dataset.require_incremental_response is True


TEST_CHAT_MESSAGES_WITH_TOOLS = [
    {
        "messages": [
            {"role": "user", "content": "Let me know the weather in Takeshiba and Izu Oshima."},
            {
                "role": "assistant",
                "content": "this is a message before tool-calling",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "dummy1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Takeshiba"},
                        },
                    },
                    {
                        "type": "function",
                        "id": "dummy2",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Izu Oshima"},
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "dummy1",
                "content": '{"weather": "sunny"}',  #  JSON string is the standard tool-response format on OpenAI API
            },
            {
                "role": "tool",
                "tool_call_id": "dummy2",
                "content": {"weather": "cloudy"},  #  For convenience, dict and list are also allowed.
            },
            {"role": "assistant", "content": "It will be sunny in Takeshiba and cloudy in Izu Oshima."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "str - city name to get weather"}},
                        "required": ["city"],
                    },
                    "return": {"type": "string", "description": "weather_info: str"},
                },
            }
        ],
    }
]


@pytest.fixture()
def mock_chat_messages_with_tools_data_path() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as f:
        for messages in TEST_CHAT_MESSAGES_WITH_TOOLS:
            f.write(json.dumps(messages) + "\n")
        f.flush()
        yield f.name


def test_load_dataset_with_tools(mock_chat_messages_with_tools_data_path: str) -> None:
    dataset = OpenAIMessagesDataset(
        file_path=mock_chat_messages_with_tools_data_path,
        message_key="messages",
        tool_definitions_key="tools",
    )

    assert len(dataset) == 1
    messages_dicts = TEST_CHAT_MESSAGES_WITH_TOOLS[0]["messages"]
    tools_dict = TEST_CHAT_MESSAGES_WITH_TOOLS[0]["tools"]
    chat_messages: dict[str, Any] = dataset[0].messages

    assert dataset[0].tools == tools_dict

    assert len(chat_messages) == 5

    # first user turn
    assert chat_messages[0] == {"role": "user", "content": messages_dicts[0]["content"]}
    # tool_calling turn
    assert chat_messages[1]["content"] == messages_dicts[1]["content"]
    input_tool_call_1 = messages_dicts[1]["tool_calls"][0]
    processed_tool_call_1 = chat_messages[1]["tool_calls"][0]
    assert processed_tool_call_1["function"]["name"] == input_tool_call_1["function"]["name"]
    assert processed_tool_call_1["function"]["arguments"] == input_tool_call_1["function"]["arguments"]
    input_tool_call_2 = messages_dicts[1]["tool_calls"][1]
    processed_tool_call_2 = chat_messages[1]["tool_calls"][1]
    assert processed_tool_call_2["function"]["name"] == input_tool_call_2["function"]["name"]
    assert processed_tool_call_2["function"]["arguments"] == input_tool_call_2["function"]["arguments"]
    # tool_results turn
    input_tool_response_1 = messages_dicts[2]
    processed_tool_response_1 = chat_messages[2]["content"]
    assert processed_tool_response_1 == input_tool_response_1["content"]
    input_tool_response_2 = messages_dicts[3]
    processed_tool_response_2 = chat_messages[3]["content"]
    assert processed_tool_response_2 == input_tool_response_2["content"]
    # assistant response turn
    assert chat_messages[4] == {"role": "assistant", "content": messages_dicts[4]["content"]}
