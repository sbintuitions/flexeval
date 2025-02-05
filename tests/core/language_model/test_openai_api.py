import os

import pytest

from flexeval import OpenAIChatAPI
from flexeval.core.language_model.openai_api import (
    message_list_from_prompt,
    prompt_from_message_list,
    remove_duplicates_from_prompt_list,
)


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def chat_lm() -> OpenAIChatAPI:
    return OpenAIChatAPI("gpt-3.5-turbo-0125")


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_log_probs(chat_lm: OpenAIChatAPI) -> None:
    prefix = "Hello, how are you?"
    text = "Good"
    log_prob = chat_lm.compute_log_probs(text, [prefix])
    assert isinstance(log_prob, None | float)
    batch_log_prob = chat_lm.batch_compute_log_probs([text], [prefix])
    assert isinstance(batch_log_prob[0], None | float)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_chat_log_probs(chat_lm: OpenAIChatAPI) -> None:
    prompt = [{"role": "user", "content": "Hello, how are you?"}]
    response = {"role": "assistant", "content": "Good"}
    log_prob = chat_lm.compute_chat_log_probs(prompt, response)
    assert isinstance(log_prob, None | float)
    batch_log_prob = chat_lm.batch_compute_chat_log_probs([prompt], [response])
    assert isinstance(batch_log_prob[0], None | float)


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
