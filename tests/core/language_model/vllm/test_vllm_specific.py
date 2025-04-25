import logging

import pytest
from transformers import AutoTokenizer

from flexeval.core.language_model import VLLM, HuggingFaceLM, LanguageModel
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


@pytest.fixture(scope="module")
def hf_lm(model_name: str = "sbintuitions/tiny-lm-chat") -> HuggingFaceLM:
    return HuggingFaceLM(
        model=model_name, model_kwargs={"torch_dtype": "float32"}, default_gen_kwargs={"temperature": 0.0}
    )


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_compute_log_probs_approximates_hf_lm(chat_lm: LanguageModel, hf_lm: HuggingFaceLM) -> None:
    prefix_list = ["それは正しい日本語ですか？"]
    text_list = ["これは正しい日本語です。"]

    vllm_log_probs = chat_lm.compute_log_probs(text_list)
    hf_log_probs = hf_lm.compute_log_probs(text_list)
    assert vllm_log_probs == pytest.approx(hf_log_probs, abs=1e-2)

    vllm_log_probs = chat_lm.compute_log_probs(text_list, prefix_list=prefix_list)
    hf_log_probs = hf_lm.compute_log_probs(text_list, prefix_list=prefix_list)
    assert vllm_log_probs == pytest.approx(hf_log_probs, abs=1e-2)


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_model_limit_tokens_generate_complete_text(chat_lm: VLLM) -> None:
    text = "Outputs numbers 0~10: 1 2 3 "
    tokenizer = AutoTokenizer.from_pretrained("sbintuitions/tiny-lm-chat")
    input_length = len(
        tokenizer(
            [text],
            add_special_tokens=False,
            return_token_type_ids=False,
        ).input_ids[0]
    )

    # if max_new_tokens only, no warnings will be sent.
    lm_output = chat_lm.complete_text(text, max_new_tokens=128)

    # if max_new_tokens > (model_limit_new_tokens = model_new_tokens - len(input_tokens)), a warning about overwriting is sent.  # noqa: E501
    chat_lm.model_limit_tokens = input_length + 3
    lm_output_limit_tokens = chat_lm.complete_text(text, max_new_tokens=128)
    assert lm_output_limit_tokens.finish_reason == "length"
    assert len(lm_output.text) > len(lm_output_limit_tokens.text)
    chat_lm.model_limit_tokens = None


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_input_length_exceeds_model_limit_new_tokens(chat_lm: VLLM, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    text = "Hello. I am a "
    tokenizer = AutoTokenizer.from_pretrained("sbintuitions/tiny-lm")
    input_length = len(
        tokenizer(
            [text],
            add_special_tokens=False,
            return_token_type_ids=False,
        ).input_ids[0]
    )
    chat_lm.model_limit_tokens = input_length
    lm_output = chat_lm.complete_text(text, max_new_tokens=128)
    assert lm_output.text == ""
    assert lm_output.finish_reason == "input_length_limit"
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("Received input that is longer than") for record in caplog.records)
    caplog.clear()
    chat_lm.model_limit_tokens = None
