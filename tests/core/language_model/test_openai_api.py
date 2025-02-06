import os

import pytest

from flexeval.core.language_model import OpenAIChatAPI, OpenAICompletionAPI


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def completion_lm() -> OpenAICompletionAPI:
    return OpenAICompletionAPI(model="gpt-3.5-turbo-instruct")


@pytest.fixture(scope="module")
def chat_lm() -> OpenAIChatAPI:
    return OpenAIChatAPI(model="gpt-3.5-turbo")


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_batch_generate_text(completion_lm: OpenAICompletionAPI) -> None:
    responses = completion_lm.batch_complete_text(
        ["質問：Completion APIってlegacyなAPIなの？"],
        max_new_tokens=20,
        stop_sequences=["。"],
    )

    assert len(responses) == 1
    assert isinstance(responses[0], str)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_batch_generate_chat_response(chat_lm: OpenAIChatAPI) -> None:
    responses = chat_lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは！"}]],
        max_new_tokens=20,
        stop_sequences=["。"],
    )

    assert len(responses) == 1
    assert isinstance(responses[0], str)
