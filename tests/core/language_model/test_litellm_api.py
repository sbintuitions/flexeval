import os

import pytest

from flexeval.core.language_model import LiteLLMChatAPI, LMOutput


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def lm() -> LiteLLMChatAPI:
    return LiteLLMChatAPI(model="gpt-4o-mini-2024-07-18")


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_batch_generate_chat_response(lm: LiteLLMChatAPI) -> None:
    responses = lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは！"}]],
        max_new_tokens=20,
        stop_sequences=["。"],
    )

    assert len(responses) == 1
    assert isinstance(responses[0], LMOutput)
    assert isinstance(responses[0].text, str)
    assert responses[0].finish_reason in {"length", "stop"}


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_chat_log_probs_for_multi_tokens(lm: LiteLLMChatAPI) -> None:
    prompt_list = [[{"role": "user", "content": "Output a number from 1 to 3."}] for _ in range(2)]
    response_list = [{"role": "assistant", "content": "1"}, {"role": "assistant", "content": "4"}]
    with pytest.raises(NotImplementedError):
        lm.batch_compute_chat_log_probs(prompt_list, response_list)
