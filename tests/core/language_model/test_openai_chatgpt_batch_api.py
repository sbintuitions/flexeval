import os

import pytest

from flexeval.core.language_model.openai_batch_api import OpenAIChatBatchAPI


def is_openai_enabled() -> bool:
    return not (os.environ.get("OPENAI_API_KEY") is None or os.environ.get("OPENAI_API_KEY") is None)


@pytest.fixture(scope="module")
def lm() -> OpenAIChatBatchAPI:
    return OpenAIChatBatchAPI(
        model="gpt-4o-mini-2024-07-18", polling_interval_seconds=6, default_gen_kwargs={"temperature": 0.7}
    )


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_create_batch_file(lm: OpenAIChatBatchAPI) -> None:
    lm.create_batch_file(
        {str(i): [[{"role": "user", "content": "こんにちは。"}]] for i in range(10)},
        max_new_tokens=40,
    )
    with open(lm.temp_jsonl_file.name) as f:
        lines = f.readlines()

    assert len(lines) == 10


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_batch_generate_chat_response(lm: OpenAIChatBatchAPI) -> None:
    responses = lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=40,
    )

    assert len(responses) == 1
    assert isinstance(responses[0], str)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_batch_compute_chat_log_probs(lm: OpenAIChatBatchAPI) -> None:
    responses = lm.batch_compute_chat_log_probs(
        [
            [{"role": "user", "content": "こんにちは。"}],
            [{"role": "user", "content": "こんにちは。"}],
            [{"role": "user", "content": "こんばんは。"}],
        ],
        [
            {"role": "user", "content": "こんにちは。"},
            {"role": "user", "content": "こんばんは。"},
            {"role": "user", "content": "こんばんは。"},
        ],
    )

    assert len(responses) == 3
    assert isinstance(responses[0], float)
