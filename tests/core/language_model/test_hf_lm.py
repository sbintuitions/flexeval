from __future__ import annotations

import functools
import logging
from typing import Any, Callable
from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from flexeval.core.language_model.base import LMOutput
from flexeval.core.language_model.hf_lm import (
    HuggingFaceLM,
    LanguageModel,
    decode_for_lm_continuation,
    get_prefix_and_completion_from_chat,
    tokenize_text_for_lm_continuation,
    tokenize_text_for_lm_prefix,
)
from tests.dummy_modules.tool_parser import DummyToolParser

from .base import BaseLanguageModelTest


@pytest.fixture(scope="module")
def lm_init_func(model: str = "sbintuitions/tiny-lm") -> Callable[..., HuggingFaceLM]:
    # use float32 because half precision is not supported in some hardware
    return functools.partial(
        HuggingFaceLM,
        model=model,
        model_kwargs={"torch_dtype": "float32"},
        tokenizer_kwargs={"use_fast": False},
        default_gen_kwargs={"do_sample": False},
    )


@pytest.fixture(scope="module")
def lm() -> HuggingFaceLM:
    # use float32 because half precision is not supported in some hardware
    return HuggingFaceLM(
        model="sbintuitions/tiny-lm",
        model_kwargs={"torch_dtype": "float32"},
        tokenizer_kwargs={"use_fast": False},
        default_gen_kwargs={"do_sample": False},
    )


@pytest.fixture(scope="module")
def chat_lm(model_name: str = "sbintuitions/tiny-lm-chat") -> HuggingFaceLM:
    return HuggingFaceLM(
        model=model_name,
        model_kwargs={"torch_dtype": "float32"},
        default_gen_kwargs={"do_sample": False},
    )


@pytest.fixture(scope="module")
def chat_lm_for_tool_calling(model_name: str = "sbintuitions/tiny-lm-chat") -> HuggingFaceLM:
    tool_parser = DummyToolParser()
    return HuggingFaceLM(
        model=model_name,
        model_kwargs={"torch_dtype": "float32"},
        default_gen_kwargs={"do_sample": False},
        tool_parser=tool_parser,
    )


@pytest.fixture(scope="module")
def chat_lm_with_system_message(model_name: str = "sbintuitions/tiny-lm-chat") -> HuggingFaceLM:
    return HuggingFaceLM(
        model=model_name,
        model_kwargs={"torch_dtype": "float32"},
        default_gen_kwargs={"do_sample": False},
        system_message="You are a helpful assistant.",
    )


@pytest.fixture(scope="module")
def chat_lm_without_system_message(model_name: str = "sbintuitions/tiny-lm-chat") -> HuggingFaceLM:
    return HuggingFaceLM(
        model=model_name,
        model_kwargs={"torch_dtype": "float32"},
        default_gen_kwargs={"do_sample": False},
    )


class TestHuggingFaceLM(BaseLanguageModelTest):
    @pytest.fixture()
    def lm(self, lm: HuggingFaceLM) -> LanguageModel:
        return lm

    @pytest.fixture()
    def chat_lm(self, chat_lm: HuggingFaceLM) -> LanguageModel:
        return chat_lm

    @pytest.fixture()
    def chat_lm_for_tool_calling(self, chat_lm_for_tool_calling: HuggingFaceLM) -> HuggingFaceLM:
        return chat_lm_for_tool_calling


@pytest.mark.parametrize(
    "tokenizer_name",
    ["rinna/japanese-gpt2-xsmall", "line-corporation/japanese-large-lm-1.7b", "tokyotech-llm/Swallow-7b-instruct-hf"],
)
def test_output_type_and_shape_from_text_for_lm_prefix(tokenizer_name: str) -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")

    for text_list in [["これは prefix です。", "こんにちは、本文です。", ""], ["", ""]]:
        model_inputs = tokenize_text_for_lm_prefix(text_list, tokenizer)

        # check the output type and shape
        assert model_inputs.input_ids.shape == model_inputs.attention_mask.shape
        assert ((model_inputs.input_ids != tokenizer.pad_token_id) == model_inputs.attention_mask).all()

        assert model_inputs.input_ids.dtype == torch.long
        assert model_inputs.attention_mask.dtype == torch.long


@pytest.mark.parametrize(
    ("tokenizer_name", "add_special_tokens", "has_bos_tokens"),
    [
        # These tokenizers do not prepend bos tokens regardless of the add_special_tokens flag
        ("rinna/japanese-gpt2-xsmall", True, False),
        ("rinna/japanese-gpt2-xsmall", False, False),
        ("line-corporation/japanese-large-lm-1.7b", True, False),
        ("line-corporation/japanese-large-lm-1.7b", False, False),
        # These tokenizers prepend bos tokens when add_special_tokens is True
        ("tokyotech-llm/Swallow-7b-instruct-hf", True, True),
        ("tokyotech-llm/Swallow-7b-instruct-hf", False, False),
    ],
)
def test_if_tokenizer_add_bos_tokens_in_an_expected_way(
    tokenizer_name: str,
    add_special_tokens: bool,
    has_bos_tokens: bool,
) -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    for text_list in [["これは prefix です。", "こんにちは、本文です。", ""], ["", ""]]:
        model_inputs = tokenize_text_for_lm_prefix(text_list, tokenizer, add_special_tokens=add_special_tokens)
        for input_ids in model_inputs.input_ids:
            assert (tokenizer.bos_token_id in input_ids) == has_bos_tokens


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "line-corporation/japanese-large-lm-1.7b",
        "rinna/japanese-gpt-1b",
        "sbintuitions/sarashina2-7b",
        "allenai/Llama-3.1-Tulu-3-8B",
    ],
)
def test_tokenize_text_for_lm_continuation(tokenizer_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    # Set pad_token for tokenizers such as "meta-llama/Meta-Llama-3-8B"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # normal test cases
    # The character 'm' forms a weird token when it follows certain multi-byte characters in Llama3 tokenizer.
    text_list = ["は続き", "is continuation.", "m"]
    batch_encoding = tokenize_text_for_lm_continuation(text_list, tokenizer)
    for i, tokens in enumerate(batch_encoding.input_ids):
        first_token = tokenizer.convert_ids_to_tokens([tokens[0]])[0]
        assert not first_token.startswith("▁")  # check if the prefix of sentencepiece is not added
        assert tokenizer.decode(tokens, skip_special_tokens=True) == text_list[i]

    # Test with conditional operations
    # This is mainly for tokenizers with add_prefix_space=True,
    # which adds a space to the beginning of the text but not to the continuation.
    text_list = ["これは文頭", "これは続き"]
    as_continuation = [False, True]
    batch_encoding = tokenize_text_for_lm_continuation(text_list, tokenizer, as_continuation=as_continuation)
    for i, (tokens, as_cont) in enumerate(zip(batch_encoding.input_ids, as_continuation)):
        first_token = tokenizer.convert_ids_to_tokens([tokens[0]])[0]
        starts_with_prefix = (not as_cont) and tokenizer.add_prefix_space
        assert first_token.startswith("▁") == starts_with_prefix
        assert tokenizer.decode(tokens, skip_special_tokens=True) == text_list[i]


@pytest.mark.parametrize(
    "tokenizer_name",
    ["sbintuitions/sarashina2-7b", "llm-jp/llm-jp-3-3.7b", "Qwen/Qwen2.5-0.5B", "allenai/Llama-3.1-Tulu-3-8B"],
)
@pytest.mark.parametrize(
    "text", ["def foo():\n", "    return 1", "こんにちは世界", "<|im_start|>Hello<|end_of_text|>Yes"]
)
def test_decode_for_lm_continuation(tokenizer_name: str, text: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    # First we need to check if the tokenizer does not change the text
    assert tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"]) == text

    # Simulate generated tokens at various text boundaries
    for i in range(1, len(text) - 1):
        prefix = text[:i]
        continuation = text[i:]
        prefix_tokens = tokenize_text_for_lm_prefix([prefix], tokenizer).input_ids[0].tolist()
        continuation_tokens = tokenize_text_for_lm_continuation([continuation], tokenizer).input_ids[0].tolist()
        prefix = tokenizer.decode(prefix_tokens, skip_special_tokens=False)
        # The point is, just decoding the continuation_tokens as follows sometimes can not restore the original text.
        # `continuation = tokenizer.decode(continuation_tokens, skip_special_tokens=True)`
        continuation = decode_for_lm_continuation(continuation_tokens, prefix_tokens, tokenizer)
        assert prefix + continuation == text


def test_if_random_seed_fixes_the_lm_outputs(lm_init_func: Callable[..., HuggingFaceLM]) -> None:
    # first check if the outputs are different without fixing the seed
    completions = set()
    for i in range(3):
        lm = lm_init_func(random_seed=i)
        completion = lm.complete_text(["<s>"], do_sample=True)[0]
        completions.add(completion.text)
    assert len(completions) > 1

    # then check if the outputs are the same with fixing the seed
    completions = set()
    for _ in range(3):
        lm = lm_init_func(random_seed=42)
        completion = lm.complete_text(["<s>"], do_sample=True)[0]
        completions.add(completion.text)
    assert len(completions) == 1

    # note that the randomness starts in __init__
    # so if you sample outputs from the same instance, the outputs will be different
    lm = lm_init_func(random_seed=42)
    completions = set()
    for _ in range(3):
        completion = lm.complete_text(["<s>"], do_sample=True)[0]
        completions.add(completion.text)
    assert len(completions) > 1


def test_if_custom_chat_template_is_given(lm_init_func: Callable[..., HuggingFaceLM]) -> None:
    # To verify that the template specified in `custom_chat_template` is passed to `tokenizer.apply_chat_template()`,
    # prepare a template where the model is expected to output "0 0..." for any input.
    custom_chat_template = "0 0 0 0 0 0 0 0 0 0 0"
    lm = lm_init_func(
        random_seed=42,
        custom_chat_template=custom_chat_template,
    )
    responses = lm.generate_chat_response([[{"role": "user", "content": "こんにちは。"}]], max_length=40)
    assert len(responses) == 1
    assert responses[0].text.strip().startswith("0 0")


@pytest.mark.parametrize(
    ("fill_with_zeros", "expected_startswith_text"),
    [
        (True, "0 0"),
        (False, "x x"),
    ],
)
def test_if_chat_template_kwargs_is_used(
    lm_init_func: Callable[..., HuggingFaceLM], fill_with_zeros: bool, expected_startswith_text: str
) -> None:
    custom_chat_template = (
        "{%- if fill_with_zeros is defined and fill_with_zeros is true -%}"
        "0 0 0 0 0 0 0 0 0 0 0"
        "{%- else -%}"
        "x x x x x x x x x x x"  # With 1 1 1, the continuation was not 1.
        "{%- endif -%}"
    )
    lm = lm_init_func(
        random_seed=42,
        custom_chat_template=custom_chat_template,
        chat_template_kwargs={"fill_with_zeros": fill_with_zeros},
    )

    responses = lm.generate_chat_response([[{"role": "user", "content": "こんにちは。"}]], max_length=40)
    assert len(responses) == 1
    assert responses[0].text.strip().startswith(expected_startswith_text)


def test_if_stop_sequences_work_as_expected(chat_lm: HuggingFaceLM) -> None:
    test_inputs = [[{"role": "user", "content": "こんにちは"}]]

    # check if the response does not have eos_token by default
    response = chat_lm.generate_chat_response(test_inputs, max_new_tokens=50)[0]
    assert response.text
    assert response.finish_reason == "stop"

    # check if ignore_eos=True works
    response = chat_lm.generate_chat_response(test_inputs, max_new_tokens=50, ignore_eos=True)[0]
    assert response.text
    assert response.finish_reason == "length"


def test_if_gen_kwargs_work_as_expected() -> None:
    lm = HuggingFaceLM(model="sbintuitions/tiny-lm", default_gen_kwargs={"max_new_tokens": 1})
    # check if the default gen_kwargs is used and the max_new_tokens is 1
    text = lm.complete_text("000000")
    assert len(text.text) == 1

    # check if the gen_kwargs will be overwritten by the given gen_kwargs
    text = lm.complete_text("000000", max_new_tokens=10)
    assert len(text.text) > 1


def test_get_prefix_and_completion_from_chat() -> None:
    tokenizer = AutoTokenizer.from_pretrained("sbintuitions/tiny-lm-chat", padding_side="left")
    prefix, completion = get_prefix_and_completion_from_chat(
        [{"role": "user", "content": "Hello."}], {"role": "assistant", "content": "Hi."}, tokenizer=tokenizer
    )
    assert prefix == "<|user|>Hello.</s><|assistant|>"
    assert completion == "Hi.</s>"

    prefix, completion = get_prefix_and_completion_from_chat(
        [{"role": "user", "content": "Hello."}],
        {"role": "assistant", "content": "Hi."},
        tokenizer=tokenizer,
        custom_chat_template="CUSTOM_TEMPLATE",
    )
    assert prefix == "CUSTOM_TEMPLATE"
    assert completion == ""


def test_model_limit_new_tokens_complete_text(lm: HuggingFaceLM) -> None:
    text = "Outputs numbers 0~10: 1 2 3 "
    tokenizer = AutoTokenizer.from_pretrained("sbintuitions/tiny-lm")
    input_length = len(
        tokenizer(
            [text],
            add_special_tokens=False,
            return_token_type_ids=False,
        ).input_ids[0]
    )

    # if max_new_tokens only, no warnings will be sent.
    lm_output = lm.complete_text(text, max_new_tokens=128)

    # if max_new_tokens > (model_limit_new_tokens = model_new_tokens - len(input_tokens)), a warning about overwriting is sent.  # noqa: E501
    lm_with_limit_tokens = HuggingFaceLM(
        model="sbintuitions/tiny-lm",
        model_kwargs={"torch_dtype": "float32"},
        tokenizer_kwargs={"use_fast": False},
        default_gen_kwargs={"do_sample": False},
        add_special_tokens=False,
        model_limit_tokens=input_length + 3,
    )
    lm_output_limit_tokens: LMOutput = lm_with_limit_tokens.complete_text(text, max_new_tokens=128)
    assert lm_output_limit_tokens.finish_reason == "length"
    assert len(lm_output.text) > len(lm_output_limit_tokens.text)


def test_if_input_length_exceeds_model_limit_new_tokens(caplog: pytest.LogCaptureFixture) -> None:
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
    lm_with_limit_tokens = HuggingFaceLM(
        model="sbintuitions/tiny-lm",
        model_kwargs={"torch_dtype": "float32"},
        tokenizer_kwargs={"use_fast": False},
        default_gen_kwargs={"do_sample": False},
        model_limit_tokens=input_length,
        add_special_tokens=False,
    )

    lm_output: LMOutput = lm_with_limit_tokens.complete_text(text, max_new_tokens=128)
    assert lm_output.text == ""
    assert lm_output.finish_reason == "input_length_limit"
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("Received input that is longer than") for record in caplog.records)
    caplog.clear()


def test_apply_chat_template_arguments_when_tools_provided(chat_lm_for_tool_calling: HuggingFaceLM) -> None:
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


def test_system_message_is_prepended_to_chat_messages(chat_lm_with_system_message: HuggingFaceLM) -> None:
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


def test_system_message_prepended_to_batch_chat_messages(chat_lm_with_system_message: HuggingFaceLM) -> None:
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
