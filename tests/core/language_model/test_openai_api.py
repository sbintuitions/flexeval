import logging
import os

import pytest
import tiktoken
from tiktoken import Encoding

from flexeval import OpenAIChatAPI
from flexeval.core.language_model.openai_api import (
    message_list_from_prompt,
    prompt_from_message_list,
    remove_duplicates_from_prompt_list,
)

model_name = "gpt-4o-mini-2024-07-18"


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def chat_lm() -> OpenAIChatAPI:
    return OpenAIChatAPI(model_name)


@pytest.fixture(scope="module")
def tokenizer() -> Encoding:
    return tiktoken.encoding_for_model(model_name)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_batch_generate_chat_response(chat_lm: OpenAIChatAPI) -> None:
    responses = chat_lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは！"}]],
        max_new_tokens=20,
        stop_sequences=["。"],
    )

    assert len(responses) == 1
    assert isinstance(responses[0], str)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_generate_chat_response(chat_lm: OpenAIChatAPI) -> None:
    response = chat_lm.generate_chat_response([{"role": "user", "content": "こんにちは。"}], max_new_tokens=40)
    assert isinstance(response, str)

    responses = chat_lm.generate_chat_response(
        [
            [{"role": "user", "content": "こんにちは。"}],
            [{"role": "user", "content": "こんばんわ"}],
        ],
        max_new_tokens=40,
    )
    assert len(responses) == 2
    assert isinstance(responses[0], str)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_batch_complete_text(chat_lm: OpenAIChatAPI) -> None:
    completions = chat_lm.batch_complete_text(["こんにちは、", "おはよう、"])
    assert len(completions) == 2
    assert isinstance(completions[0], str)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_complete_text(chat_lm: OpenAIChatAPI) -> None:
    completion = chat_lm.complete_text("こんにちは、")
    assert isinstance(completion, str)

    completions = chat_lm.batch_complete_text(["こんにちは、", "おはよう、"])
    assert len(completions) == 2
    assert isinstance(completions[0], str)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_max_tokens(chat_lm: OpenAIChatAPI, tokenizer: Encoding) -> None:
    # enter prompts where a long output is expected.
    completion = chat_lm.batch_complete_text(["47都道府県をカンマ区切りで列挙してください。"], max_new_tokens=1)[0]
    assert len(tokenizer.encode(completion)) == 1


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_stop_sequences(chat_lm: OpenAIChatAPI) -> None:
    completion = chat_lm.batch_complete_text(
        ["10進数の2は2進数で表すと何になりますか？回答のみ出力してください。"],
        stop_sequences=["1"],
        max_new_tokens=10,
        temperature=0.0,
    )[0]
    assert completion.strip() == ""

    completion = chat_lm.batch_complete_text(
        ["10進数の2は2進数で表すと何になりますか？回答のみ出力してください。"],
        stop_sequences=["0"],
        max_new_tokens=10,
        temperature=0.0,
    )[0]
    assert completion.strip() == "1"


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_warning_if_conflict_max_new_tokens(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    chat_lm_with_max_new_tokens = OpenAIChatAPI(model_name, default_gen_kwargs={"max_completion_tokens": 10})
    chat_lm_with_max_new_tokens.batch_generate_chat_response(
        [[{"role": "user", "content": "テスト"}]], max_new_tokens=20
    )
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("You specified both `max_new_tokens`") for record in caplog.records)


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
