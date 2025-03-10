import pytest

from flexeval.core.language_model.base import LMOutput
from flexeval.core.language_model.vllm_model import VLLM, LanguageModel
from tests.conftest import is_vllm_enabled


@pytest.fixture(scope="module")
def chat_lm() -> VLLM:
    llm = VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
        },
        tokenizer_kwargs={"use_fast": False},
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_stop_sequences_work_as_expected(chat_lm: VLLM) -> None:
    test_inputs = [[{"role": "user", "content": "こんにちは"}]]

    # check if the response does not have eos_token by default
    response = chat_lm.batch_generate_chat_response(test_inputs, max_new_tokens=50)[0]
    assert response.text
    assert response.finish_reason == "stop"

    # check if ignore_eos=True works
    response = chat_lm.batch_generate_chat_response(test_inputs, max_new_tokens=50, ignore_eos=True)[0]
    assert response.text
    assert response.finish_reason == "length"


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_compute_chat_log_probs(chat_lm: VLLM) -> None:
    log_probs_natural = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Hello, how are you?"}]],
        [{"role": "assistant", "content": "Good."}],
    )
    log_probs_unnatural_lang = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Hello, how are you?"}]],
        [{"role": "assistant", "content": "!?本日は晴天ナリ."}],
    )
    log_probs_unnatural_ord = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Good."}]],
        [{"role": "assistant", "content": "Hello, how are you?"}],
    )

    assert len(log_probs_natural) == 1
    assert isinstance(log_probs_natural[0], float)
    assert len(log_probs_unnatural_lang) == 1
    assert isinstance(log_probs_unnatural_lang[0], float)
    assert len(log_probs_unnatural_ord) == 1
    assert isinstance(log_probs_unnatural_ord[0], float)
    assert log_probs_natural[0] > log_probs_unnatural_lang[0]
    assert log_probs_natural[0] > log_probs_unnatural_ord[0]


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_compute_chat_log_probs(chat_lm: VLLM) -> None:
    prompt = [{"role": "user", "content": "Hello, how are you?"}]
    response = {"role": "assistant", "content": "Good."}
    log_prob = chat_lm.compute_chat_log_probs(prompt, response)
    assert isinstance(log_prob, float)
    batch_log_prob = chat_lm.batch_compute_chat_log_probs([prompt], [response])
    assert log_prob == batch_log_prob[0]


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_generate_chat_response(chat_lm: LanguageModel) -> None:
    responses = chat_lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=40,
    )
    assert len(responses) == 1
    assert isinstance(responses[0], LMOutput)
    assert isinstance(responses[0].text, str)
    assert isinstance(responses[0].finish_reason, str)
