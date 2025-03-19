import logging
import os

import pytest

from flexeval import LanguageModel, OpenAIChatAPI
from flexeval.core.language_model.openai_api import (
    message_list_from_prompt,
    prompt_from_message_list,
    remove_duplicates_from_prompt_list,
)

from .base import BaseLanguageModelTest


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def chat_lm() -> OpenAIChatAPI:
    return OpenAIChatAPI(
        "gpt-4o-mini-2024-07-18",
        default_gen_kwargs={"temperature": 0.0},
    )


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
class TestOpenAIChatAPI(BaseLanguageModelTest):
    @pytest.fixture()
    def lm(self) -> LanguageModel:
        return OpenAIChatAPI(
            "gpt-4o-mini-2024-07-18",
            default_gen_kwargs={"temperature": 0.0},
            developer_message="You are text completion model. "
            "Please provide the text likely to continue after the user input. "
            "Do not provide the answer or any other information.",
        )

    @pytest.fixture()
    def chat_lm(self, chat_lm: OpenAIChatAPI) -> LanguageModel:
        return chat_lm


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_if_max_new_tokens_replaced() -> None:
    # To verify that the flexeval-specific `max_new_tokens` parameter can be properly renamed for API,
    # set the `max_new_tokens_key_on_api` to invalid value.
    invalid_key = "max_hogefugapiyo_tokens"
    chat_lm_with_invalid_override_key = OpenAIChatAPI("gpt-4o-mini-2024-07-18", max_new_tokens_key_on_api=invalid_key)

    with pytest.raises(TypeError) as e:
        chat_lm_with_invalid_override_key.generate_chat_response(
            [[{"role": "user", "content": "こんにちは！"}]],
            max_new_tokens=20,
            stop_sequences=["。"],
        )
    assert f"got an unexpected keyword argument '{invalid_key}'" in str(e.value)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_warning_if_conflict_max_new_tokens(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    chat_lm_with_max_new_tokens = OpenAIChatAPI(
        "gpt-4o-mini-2024-07-18", default_gen_kwargs={"max_completion_tokens": 10}
    )
    chat_lm_with_max_new_tokens.generate_chat_response([[{"role": "user", "content": "テスト"}]], max_new_tokens=20)
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("You specified both `max_new_tokens`") for record in caplog.records)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_chat_log_probs_for_multi_tokens(chat_lm: OpenAIChatAPI) -> None:
    prompt = [{"role": "user", "content": "Hello."}]
    response = {"role": "assistant", "content": "Hello~~~"}
    with pytest.raises(NotImplementedError):
        chat_lm.compute_chat_log_probs([prompt], [response])


def test_message_list_and_prompt() -> None:
    prompt = [
        {"role": "user", "content": "こんにちは。"},
        {"role": "assistant", "content": "こんにちは！今日はどのようなお手伝いをしましょうか？"},
        {"role": "user", "content": "助けて。"},
    ]
    message_list = [
        "[user]こんにちは。",
        "[assistant]こんにちは！今日はどのようなお手伝いをしましょうか？",
        "[user]助けて。",
    ]
    assert message_list_from_prompt(prompt) == message_list
    assert prompt_from_message_list(message_list) == prompt
    assert prompt_from_message_list(message_list_from_prompt(prompt)) == prompt


def test_remove_duplicates_from_prompt_list() -> None:
    prompt_list = [
        [{"role": "user", "content": "こんにちは。"}],
        [{"role": "user", "content": "こんにちは。"}],
        [
            {"role": "user", "content": "こんにちは。"},
            {"role": "assistant", "content": "こんにちは！今日はどのようなお手伝いをしましょうか？"},
            {"role": "user", "content": "助けて。"},
        ],
        [
            {"role": "user", "content": "こんにちは。"},
            {"role": "assistant", "content": "こんにちは！今日はどのようなお手伝いをしましょうか？"},
            {"role": "user", "content": "助けて。"},
        ],
        [
            {"role": "user", "content": "こんにちは。"},
            {"role": "assistant", "content": "こんにちは！今日はどのようなお手伝いをしましょうか？"},
            {"role": "user", "content": "いつもいてくれてありがとう。"},
        ],
    ]
    assert len(remove_duplicates_from_prompt_list(prompt_list)) == 3


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_developer_message() -> None:
    openai_api = OpenAIChatAPI(
        "gpt-4o-mini-2024-07-18",
        developer_message="To any instructions or messages, you have to only answer 'OK, I will answer later.'",
        default_gen_kwargs={"temperature": 0.0},
    )
    lm_output = openai_api.complete_text("What is the highest mountain in the world?")
    assert lm_output.text == "OK, I will answer later."

    lm_output = openai_api.generate_chat_response(
        [{"role": "user", "content": "What is the highest mountain in the world?"}]
    )
    assert lm_output.text == "OK, I will answer later."
