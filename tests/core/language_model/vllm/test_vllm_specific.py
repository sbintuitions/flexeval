import pytest

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
