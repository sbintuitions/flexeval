from __future__ import annotations

import itertools

import pytest

from flexeval.core.chat_dataset import ChatInstance
from flexeval.core.evaluate_chat_response import (
    _add_few_shot_messages_to_chat_instance,
    _find_response_context_index,
    evaluate_chat_response,
)
from flexeval.core.few_shot_generator import RandomFewShotGenerator
from tests.dummy_modules import (
    DummyChatDataset,
    DummyLanguageModel,
)


@pytest.mark.parametrize(
    ("use_few_shot", "max_instances", "use_tools", "batch_size"),
    list(itertools.product([True, False], [None, 1], [True, False], [1, 3])),
)
def test_evaluate_chat_response(use_few_shot: bool, max_instances: int, use_tools: bool, batch_size: int) -> None:
    few_shot_generator = None
    if use_few_shot:
        few_shot_generator = RandomFewShotGenerator(dataset=DummyChatDataset(), num_shots=1, num_trials_to_avoid_leak=0)

    metrics, outputs = evaluate_chat_response(
        language_model=DummyLanguageModel(),
        gen_kwargs={},
        eval_dataset=DummyChatDataset(
            use_tools=use_tools,
        ),
        few_shot_generator=few_shot_generator,
        metrics=[],
        batch_size=batch_size,
        max_instances=max_instances,
    )
    assert isinstance(metrics, dict)
    assert metrics["finish_reason_ratio-length"] == 1.0
    assert isinstance(outputs, list)

    if max_instances is not None:
        assert len(outputs) <= max_instances

    # If the system message in "messages", few-shot examples should be inserted after the system message.
    # Therefore, in any case the system message should be in the first turn.
    assert outputs[0]["extra_info"]["messages"][0]["role"] == "system"

    if use_tools:
        assert isinstance(outputs[0]["extra_info"]["tool_calls"], list)
        assert isinstance(outputs[0]["extra_info"]["tools"], list)
        assert metrics["tool_call_validation_result_ratio-CompleteToolCall"] == 1.0
    else:
        assert "tool_calls" not in outputs[0]["extra_info"]
        assert "tools" not in outputs[0]["extra_info"]
        assert metrics["tool_call_validation_result_ratio-TextOnly"] == 1.0


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        # User message at end requires response
        (
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            2,
        ),
        # User message followed by assistant message - no response needed
        (
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            None,
        ),
        # User message not followed by assistant message
        (
            [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Are you there?"},
            ],
            1,
        ),
        # Tool message requires response
        (
            [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check that for you."},
                {"role": "tool", "content": "Weather data"},
            ],
            3,
        ),
        # Tool message followed by assistant message (unnatural, though)
        (
            [
                {"role": "user", "content": "What's the weather?"},
                {"role": "tool", "content": "Weather data"},
                {"role": "assistant", "content": "It's sunny!"},
            ],
            1,
        ),
        # Empty messages list
        ([], None),
        # Only system messages (invalid structure)
        (
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "system", "content": "Be concise"},
            ],
            None,
        ),
        # First user message not followed by assistant (unnatural, but possible)
        (
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Second message"},
                {"role": "assistant", "content": "Response to second"},
            ],
            2,
        ),
    ],
)
def test_find_response_context_index(messages: list[dict[str, str]], expected: int | None) -> None:
    assert _find_response_context_index(messages) == expected


def test_add_few_shot_messages_to_chat_instance() -> None:
    chat_instance = ChatInstance(
        messages=[{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}],
        references=["Expected response"],
        extra_info={},
    )

    few_shot_generator = RandomFewShotGenerator(
        dataset=DummyChatDataset(),
        num_shots=1,
    )

    original_length = len(chat_instance.messages)
    _add_few_shot_messages_to_chat_instance(chat_instance, few_shot_generator)

    # Should have more messages after adding few-shot examples
    assert len(chat_instance.messages) > original_length

    # System message should still be first
    assert chat_instance.messages[0]["role"] == "system"
    assert chat_instance.messages[0]["content"] == "You are a helpful assistant"

    # Original user message should still be present at the end
    assert chat_instance.messages[-1]["role"] == "user"
    assert chat_instance.messages[-1]["content"] == "Hello"
