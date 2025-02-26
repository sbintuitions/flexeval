import logging
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
@pytest.mark.batch_api()
def test_create_batch_file(lm: OpenAIChatBatchAPI) -> None:
    lm.create_batch_file(
        {str(i): [[{"role": "user", "content": "こんにちは。"}]] for i in range(10)},
        max_new_tokens=40,
    )
    with open(lm.temp_jsonl_file.name) as f:
        lines = f.readlines()

    assert len(lines) == 10


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
@pytest.mark.batch_api()
def test_batch_generate_chat_response(lm: OpenAIChatBatchAPI) -> None:
    responses = lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=40,
    )

    assert len(responses) == 1
    assert isinstance(responses[0], str)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_warning_if_conflict_max_new_tokens(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    chat_lm_with_max_new_tokens = OpenAIChatBatchAPI(
        "[dummy_model]", #  to avoid long waiting time, set a dummy model name to cause an error.
        default_gen_kwargs={"max_completion_tokens": 10},
    )
    with pytest.raises(ValueError):
        chat_lm_with_max_new_tokens.batch_generate_chat_response(
            [[{"role": "user", "content": "テスト"}]], max_new_tokens=20
        )
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("You specified both `max_new_tokens`") for record in caplog.records)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_batch_compute_chat_log_probs(lm: OpenAIChatBatchAPI) -> None:
    prompt_list = [[{"role": "user", "content": "Output a number from 1 to 3."}] for _ in range(2)]
    response_list = [{"role": "assistant", "content": "1"}, {"role": "assistant", "content": "4"}]
    log_probs = lm.batch_compute_chat_log_probs(prompt_list, response_list)
    assert isinstance(log_probs, list)
    assert log_probs[0] > log_probs[1] or 0


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_chat_log_probs_for_multi_tokens(lm: OpenAIChatBatchAPI) -> None:
    prompt = [{"role": "user", "content": "Hello."}]
    response = {"role": "assistant", "content": "Hello~~~"}
    with pytest.raises(NotImplementedError):
        lm.batch_compute_chat_log_probs([prompt], [response])
