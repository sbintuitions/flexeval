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


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_model_limit_max_tokens_generate_chat_response(
    chat_lm: OpenAIChatAPI, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    messages = [{"role": "user", "content": "Hello."}]

    # if max_new_tokens only, no warnings will be sent.
    chat_lm.generate_chat_response(messages, max_new_tokens=128)
    assert len(caplog.records) == 0

    # if max_new_tokens > model_limit_completion_tokens, a warning about overwriting is sent.
    chat_lm_with_limit_tokens = OpenAIChatAPI("gpt-4o-mini-2024-07-18", model_limit_new_tokens=1)
    chat_lm_with_limit_tokens.generate_chat_response(messages, max_new_tokens=128)
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("The specified `max_new_tokens` (128) exceeds") for record in caplog.records)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_model_limit_max_tokens_complete_text(chat_lm: OpenAIChatAPI, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    text = "Hello."

    # if max_new_tokens only, no warnings will be sent.
    chat_lm.complete_text(text, max_new_tokens=128)
    assert len(caplog.records) == 0
    caplog.clear()

    # if max_new_tokens > model_limit_new_tokens, a warning about overwriting is sent.
    chat_lm_with_limit_tokens = OpenAIChatAPI("gpt-4o-mini-2024-07-18", model_limit_new_tokens=1)
    chat_lm_with_limit_tokens.complete_text(text, max_new_tokens=128)
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("The specified `max_new_tokens` (128) exceeds") for record in caplog.records)
    caplog.clear()
