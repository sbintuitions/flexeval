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
def test_compute_chat_log_probs(chat_lm: OpenAIChatAPI) -> None:
    prompt = [{"role": "user", "content": "Output a number from 1 to 3."}]
    response = {"role": "assistant", "content": "1"}
    log_prob = chat_lm.compute_chat_log_probs(prompt, response)
    assert isinstance(log_prob, float)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_batch_compute_chat_log_probs(chat_lm: OpenAIChatAPI) -> None:
    prompt_list = [[{"role": "user", "content": "Output a number from 1 to 3."}] for _ in range(2)]
    response_list = [{"role": "assistant", "content": "1"}, {"role": "assistant", "content": "4"}]
    log_probs = chat_lm.batch_compute_chat_log_probs(prompt_list, response_list)
    assert isinstance(log_probs, list)
    assert log_probs[0] > log_probs[1] or 0


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_chat_log_probs_for_multi_tokens(chat_lm: OpenAIChatAPI) -> None:
    prompt = [{"role": "user", "content": "Hello."}]
    response = {"role": "assistant", "content": "Hello~~~"}
    with pytest.raises(NotImplementedError):
        chat_lm.batch_compute_chat_log_probs([prompt], [response])


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
