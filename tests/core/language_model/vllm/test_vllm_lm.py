import pytest

from flexeval.core.language_model.base import LMOutput
from flexeval.core.language_model.hf_lm import HuggingFaceLM
from flexeval.core.language_model.vllm_model import VLLM, LanguageModel
from tests.conftest import is_vllm_enabled


@pytest.fixture(scope="module")
def lm() -> VLLM:
    llm = VLLM(
        model="sbintuitions/tiny-lm",
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
def hf_lm() -> HuggingFaceLM:
    return HuggingFaceLM(
        model="sbintuitions/tiny-lm",
        model_kwargs={"torch_dtype": "float32"},
        tokenizer_kwargs={"use_fast": False},
    )


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_complete_text(lm: LanguageModel) -> None:
    completions = lm.batch_complete_text(["こんにちは、", "おはよう、"])
    assert len(completions) == 2
    assert isinstance(completions[0], LMOutput)
    assert isinstance(completions[0].text, str)
    assert isinstance(completions[0].finish_reason, str)


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_complete_text_is_not_affected_by_batch(lm: LanguageModel) -> None:
    single_batch_input = ["こんにちは。今日もいい天気。"]
    multi_batch_inputs = ["こんにちは。今日もいい天気。", "Lorem ipsum dolor sit amet, "]

    gen_kwargs = {"stop_sequences": ["。"], "max_new_tokens": 100}
    completions_without_batch = lm.batch_complete_text(single_batch_input, **gen_kwargs)
    completions_with_batch = lm.batch_complete_text(multi_batch_inputs, **gen_kwargs)
    assert completions_without_batch[0].text == completions_with_batch[0].text
    assert completions_without_batch[0].finish_reason == completions_with_batch[0].finish_reason


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_max_tokens(lm: LanguageModel) -> None:
    # assume that the lm will repeat 0
    completion = lm.batch_complete_text(["0 0 0 0 0 0 0 0 0 0"], max_new_tokens=1)[0]
    assert len(completion.text.strip()) == 1
    assert completion.finish_reason == "length"


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_stop_sequences(lm: LanguageModel) -> None:
    # assume that the lm will repeat "10"
    completion = lm.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["1"], max_new_tokens=10)[0]
    assert completion.text.strip() == ""
    assert completion.finish_reason == "stop"

    completion = lm.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["0"], max_new_tokens=10)[0]
    assert completion.text.strip() == "1"
    assert completion.finish_reason == "stop"


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_compute_log_probs(lm: LanguageModel) -> None:
    log_prob = lm.compute_log_probs("こんにちは")
    assert isinstance(log_prob, float)

    log_probs = lm.batch_compute_log_probs(["こんにちは", "こんばんは"])
    assert len(log_probs) == 2
    assert isinstance(log_probs[0], float)


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_compute_log_probs_produces_reasonable_comparisons(lm: LanguageModel) -> None:
    # test if the shorter sentence has higher log prob
    log_probs = lm.batch_compute_log_probs(["これは正しい日本語です。", "これは正しい日本語です。そして…"])
    assert log_probs[0] > log_probs[1]

    # test if the more natural short phrase has higher log prob
    log_probs = lm.batch_compute_log_probs(["こんにちは", "コニチハ"])
    assert log_probs[0] > log_probs[1]

    # test if the grammatical sentence has higher log prob
    log_probs = lm.batch_compute_log_probs(["これは正しい日本語です。", "は正いしこれで日語本す。"])
    assert log_probs[0] > log_probs[1]

    # test if the right prefix reduces the log prob
    log_probs = lm.batch_compute_log_probs(["富士山", "富士山"], prefix_list=["日本で一番高い山は", "Yes, we are"])
    assert log_probs[0] > log_probs[1]


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_compute_log_probs_approximates_hf_lm(lm: LanguageModel, hf_lm: HuggingFaceLM) -> None:
    prefix_list = ["それは正しい日本語ですか？"]
    text_list = ["これは正しい日本語です。"]

    vllm_log_probs = lm.batch_compute_log_probs(text_list)
    hf_log_probs = hf_lm.batch_compute_log_probs(text_list)
    assert vllm_log_probs == pytest.approx(hf_log_probs, abs=1e-2)

    vllm_log_probs = lm.batch_compute_log_probs(text_list, prefix_list=prefix_list)
    hf_log_probs = hf_lm.batch_compute_log_probs(text_list, prefix_list=prefix_list)
    assert vllm_log_probs == pytest.approx(hf_log_probs, abs=1e-2)
