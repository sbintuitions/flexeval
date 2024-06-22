import pytest

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
        model_name="sbintuitions/tiny-lm",
        model_kwargs={"seed": 42, "gpu_memory_utilization": 0.1, "enforce_eager": True},
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
        model_name="sbintuitions/tiny-lm-chat",
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
        model_name="sbintuitions/tiny-lm-chat",
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
