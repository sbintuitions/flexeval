from __future__ import annotations

import logging
from typing import Any, Callable, Generator
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer

from flexeval.core.language_model import VLLM, HuggingFaceLM
from tests.conftest import is_vllm_enabled
from tests.dummy_modules.tool_parser import DummyToolParser


@pytest.fixture(scope="module")
def chat_lm() -> Generator[VLLM, None, None]:
    llm = VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "dtype": "float32",
            "disable_custom_all_reduce": True,
        },
        tokenizer_kwargs={"use_fast": False},
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def chat_lm_qwen() -> Generator[VLLM, None, None]:
    llm = VLLM(
        model="Qwen/Qwen3-0.6B-Base",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
        },
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def chat_lm_for_tool_calling() -> Generator[VLLM, None, None]:
    tool_parser = DummyToolParser()
    llm = VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "dtype": "float32",
            "disable_custom_all_reduce": True,
        },
        tokenizer_kwargs={"use_fast": False},
        tool_parser=tool_parser,
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def hf_lm(model_name: str = "sbintuitions/tiny-lm-chat") -> HuggingFaceLM:
    return HuggingFaceLM(
        model=model_name, model_kwargs={"torch_dtype": "float32"}, default_gen_kwargs={"temperature": 0.0}
    )


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
@pytest.mark.parametrize("chat_lm_name", ["chat_lm", "chat_lm_qwen"])
def test_batch_compute_log_probs_approximates_hf_lm(
    request: pytest.FixtureRequest,
    chat_lm_name: str,
    hf_lm: HuggingFaceLM,
) -> None:
    chat_lm = request.getfixturevalue(chat_lm_name)
    prefix_list = ["それは正しい日本語ですか？"]
    text_list = ["これは正しい日本語です。"]

    vllm_log_probs = chat_lm.compute_log_probs(text_list)
    hf_log_probs = hf_lm.compute_log_probs(text_list)
    assert vllm_log_probs == pytest.approx(hf_log_probs, abs=1e-2)

    vllm_log_probs = chat_lm.compute_log_probs(text_list, prefix_list=prefix_list)
    hf_log_probs = hf_lm.compute_log_probs(text_list, prefix_list=prefix_list)
    assert vllm_log_probs == pytest.approx(hf_log_probs, abs=1e-2)


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_model_limit_tokens_generate_complete_text(chat_lm: VLLM) -> None:
    text = "Outputs numbers 0~10: 1 2 3 "
    tokenizer = AutoTokenizer.from_pretrained("sbintuitions/tiny-lm-chat")
    input_length = len(
        tokenizer(
            [text],
            add_special_tokens=False,
            return_token_type_ids=False,
        ).input_ids[0]
    )

    # if max_new_tokens only, no warnings will be sent.
    lm_output = chat_lm.complete_text(text, max_new_tokens=128)

    # if max_new_tokens > (model_limit_new_tokens = model_new_tokens - len(input_tokens)), a warning about overwriting is sent.  # noqa: E501
    chat_lm.model_limit_tokens = input_length + 3
    lm_output_limit_tokens = chat_lm.complete_text(text, max_new_tokens=128)
    assert lm_output_limit_tokens.finish_reason == "length"
    assert len(lm_output.text) > len(lm_output_limit_tokens.text)
    chat_lm.model_limit_tokens = None


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_input_length_exceeds_model_limit_new_tokens(chat_lm: VLLM, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    text = "Hello. I am a "
    tokenizer = AutoTokenizer.from_pretrained("sbintuitions/tiny-lm")
    input_length = len(
        tokenizer(
            [text],
            add_special_tokens=False,
            return_token_type_ids=False,
        ).input_ids[0]
    )
    chat_lm.model_limit_tokens = input_length
    lm_output = chat_lm.complete_text(text, max_new_tokens=128)
    assert lm_output.text == ""
    assert lm_output.finish_reason == "input_length_limit"
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("Received input that is longer than") for record in caplog.records)
    caplog.clear()
    chat_lm.model_limit_tokens = None


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_apply_chat_template_arguments_when_tools_provided(chat_lm_for_tool_calling: VLLM) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for provided city in celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    chat_messages = [{"role": "user", "content": "What's the weather like in Paris today?"}]
    with patch.object(chat_lm_for_tool_calling.tokenizer, "apply_chat_template", return_value="text") as mock_method:
        chat_lm_for_tool_calling.generate_chat_response(chat_messages, max_new_tokens=1)
        args, kwargs = mock_method.call_args
        assert args[0] == chat_messages
        assert kwargs["tools"] is None

    with patch.object(chat_lm_for_tool_calling.tokenizer, "apply_chat_template", return_value="text") as mock_method:
        chat_lm_for_tool_calling.generate_chat_response(chat_messages, tools=tools, max_new_tokens=1)
        args, kwargs = mock_method.call_args
        assert args[0] == chat_messages
        assert kwargs["tools"] == tools


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_system_message_is_prepended_to_chat_messages(chat_lm_with_system_message: VLLM) -> None:
    """Test that system message is prepended to chat messages in generate_chat_response."""
    chat_messages = [{"role": "user", "content": "Hello"}]

    # Mock the tokenizer's apply_chat_template to capture the messages
    original_apply_chat_template = chat_lm_with_system_message.tokenizer.apply_chat_template
    captured_messages = None

    def mock_apply_chat_template(messages: list[list[dict[str, Any]]], **kwargs) -> Callable:
        nonlocal captured_messages
        captured_messages = messages
        return original_apply_chat_template(messages, **kwargs)

    chat_lm_with_system_message.tokenizer.apply_chat_template = mock_apply_chat_template

    try:
        chat_lm_with_system_message.generate_chat_response(chat_messages, max_new_tokens=1)

        # Check that system message was prepended
        assert len(captured_messages) == 2
        assert captured_messages[0]["role"] == "system"
        assert captured_messages[0]["content"] == "You are a helpful assistant."
        assert captured_messages[1]["role"] == "user"
        assert captured_messages[1]["content"] == "Hello"
    finally:
        # Restore original method
        chat_lm_with_system_message.tokenizer.apply_chat_template = original_apply_chat_template


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_system_message_prepended_to_batch_chat_messages(chat_lm_with_system_message: VLLM) -> None:
    """Test that system message is prepended to each conversation in batch generate_chat_response."""
    chat_messages_list = [[{"role": "user", "content": "Hello"}], [{"role": "user", "content": "Hi there"}]]

    # Mock the tokenizer's apply_chat_template to capture the messages
    original_apply_chat_template = chat_lm_with_system_message.tokenizer.apply_chat_template
    captured_messages_list = []

    def mock_apply_chat_template(messages: list[list[dict[str, Any]]], **kwargs) -> Callable:
        captured_messages_list.append(messages.copy())
        return original_apply_chat_template(messages, **kwargs)

    chat_lm_with_system_message.tokenizer.apply_chat_template = mock_apply_chat_template

    try:
        chat_lm_with_system_message.generate_chat_response(chat_messages_list, max_new_tokens=1)

        # Check that system message was prepended to both conversations
        assert len(captured_messages_list) == 2

        for captured_messages in captured_messages_list:
            assert len(captured_messages) == 2
            assert captured_messages[0]["role"] == "system"
            assert captured_messages[0]["content"] == "You are a helpful assistant."
            assert captured_messages[1]["role"] == "user"
    finally:
        # Restore original method
        chat_lm_with_system_message.tokenizer.apply_chat_template = original_apply_chat_template


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_set_random_seed(chat_lm: VLLM):
    chat_lm.set_random_seed(42)
    assert chat_lm.default_gen_kwargs["seed"] == 42
