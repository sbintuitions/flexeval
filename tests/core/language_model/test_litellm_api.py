import os

import pytest

from flexeval.core.language_model import LiteLLMChatAPI


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def lm() -> LiteLLMChatAPI:
    return LiteLLMChatAPI(model="openai/gpt-3.5-turbo")


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_batch_generate_chat_response(lm: LiteLLMChatAPI) -> None:
    responses = lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは！"}]],
        max_new_tokens=20,
        stop_sequences=["。"],
    )

    assert len(responses) == 1
    assert isinstance(responses[0], str)
