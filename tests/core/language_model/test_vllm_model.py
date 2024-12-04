import pytest

from flexeval.core.language_model.hf_lm import HuggingFaceLM
from flexeval.core.language_model.vllm_model import VLLM, LanguageModel


def is_vllm_enabled() -> bool:
    try:
        import torch
        import vllm  # noqa: F401

        return torch.cuda.device_count() > 0
    except ImportError:
        return False


@pytest.fixture(scope="module")
def lm() -> VLLM:
    return VLLM(
        model="sbintuitions/tiny-lm",
        model_kwargs={"seed": 42, "gpu_memory_utilization": 0.1, "enforce_eager": True},
        tokenizer_kwargs={"use_fast": False},
    )


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
    assert isinstance(completions[0], str)


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_complete_text_is_not_affected_by_batch(lm: LanguageModel) -> None:
    single_batch_input = ["こんにちは。今日もいい天気。"]
    multi_batch_inputs = ["こんにちは。今日もいい天気。", "Lorem ipsum dolor sit amet, "]

    gen_kwargs = {"stop_sequences": ["。"], "max_new_tokens": 100}
    completions_without_batch = lm.batch_complete_text(single_batch_input, **gen_kwargs)
    completions_with_batch = lm.batch_complete_text(multi_batch_inputs, **gen_kwargs)
    assert completions_without_batch[0] == completions_with_batch[0]


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_max_tokens(lm: LanguageModel) -> None:
    # assume that the lm will repeat 0
    completion = lm.batch_complete_text(["0 0 0 0 0 0 0 0 0 0"], max_new_tokens=1)[0]
    assert len(completion.strip()) == 1


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_stop_sequences(lm: LanguageModel) -> None:
    # assume that the lm will repeat "10"
    completion = lm.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["1"], max_new_tokens=10)[0]
    assert completion.strip() == ""

    completion = lm.batch_complete_text(["10 10 10 10 10 10 "], stop_sequences=["0"], max_new_tokens=10)[0]
    assert completion.strip() == "1"


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


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_generate_chat_response(lm: LanguageModel) -> None:
    responses = lm.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=40,
    )
    assert len(responses) == 1
    assert isinstance(responses[0], str)


@pytest.fixture(scope="module")
def chat_lm_with_custom_chat_template() -> VLLM:
    """
    We initialize VLLM in a fixture.
    Otherwise, tests will fail with the following error:
    ```
    AssertionError: Error in memory profiling.
    This happens when the GPU memory was not properly cleaned up before initializing the vLLM instance.
    ```
    """

    # To verify that the template specified in `custom_chat_template` is passed to `tokenizer.apply_chat_template()`,
    # prepare a template where the model is expected to output "0 0..." for any input.
    custom_chat_template = "0 0 0 0 0 0 0 0 0 0 0"
    return VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={"seed": 42, "gpu_memory_utilization": 0.1, "enforce_eager": True},
        tokenizer_kwargs={"use_fast": False},
        custom_chat_template=custom_chat_template,
    )


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_custom_chat_template_is_given(chat_lm_with_custom_chat_template: VLLM) -> None:
    responses = chat_lm_with_custom_chat_template.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=10,
    )
    assert len(responses) == 1
    assert responses[0].strip().startswith("0 0")


@pytest.fixture(scope="module")
def chat_lm() -> VLLM:
    return VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={"seed": 42, "gpu_memory_utilization": 0.1, "enforce_eager": True},
        tokenizer_kwargs={"use_fast": False},
    )


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_stop_sequences_work_as_expected(chat_lm: VLLM) -> None:
    test_inputs = [[{"role": "user", "content": "こんにちは"}]]
    eos_token = "</s>"  # noqa: S105

    # check if the response does not have eos_token by default
    response = chat_lm.batch_generate_chat_response(test_inputs, max_new_tokens=50)[0]
    assert not response.endswith(eos_token)

    # check if the response has eos_token with include_stop_str_in_output=True
    response = chat_lm.batch_generate_chat_response(test_inputs, max_new_tokens=50, include_stop_str_in_output=True)[0]
    assert response.endswith(eos_token)

    # check if ignore_eos=True works
    response = chat_lm.batch_generate_chat_response(test_inputs, max_new_tokens=50, ignore_eos=True)[0]
    assert eos_token in response[: -len(eos_token)]


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_gen_kwargs_work_as_expected() -> None:
    lm = VLLM(model="sbintuitions/tiny-lm", default_gen_kwargs={"max_new_tokens": 1})
    # check if the default gen_kwargs is used and the max_new_tokens is 1
    text = lm.complete_text("000000")
    assert len(text) == 1

    # check if the gen_kwargs will be overwritten by the given gen_kwargs
    text = lm.complete_text("000000", max_new_tokens=10)
    assert len(text) > 1


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_batch_compute_chat_log_probs(chat_lm: HuggingFaceLM) -> None:
    log_probs_natural = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Hello, how are you?"}]],
        [{"role": "assistant", "content": "Good."}],
    )
    log_probs_unnatural = chat_lm.batch_compute_chat_log_probs(
        [[{"role": "user", "content": "Hello, how are you?"}]],
        [{"role": "assistant", "content": "!?本日は晴天ナリ."}],
    )

    assert len(log_probs_natural) == 1
    assert isinstance(log_probs_natural[0], float)
    assert len(log_probs_unnatural) == 1
    assert isinstance(log_probs_unnatural[0], float)
    assert log_probs_natural[0] > log_probs_unnatural[0]


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_compute_chat_log_probs(chat_lm: HuggingFaceLM) -> None:
    prompt = [{"role": "user", "content": "Hello, how are you?"}]
    response = {"role": "assistant", "content": "Good."}
    log_prob = chat_lm.compute_chat_log_probs(prompt, response)
    assert isinstance(log_prob, float)
    batch_log_prob = chat_lm.batch_compute_chat_log_probs([prompt], [response])
    assert log_prob == batch_log_prob[0]
