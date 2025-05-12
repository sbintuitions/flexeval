import pytest

from flexeval.core.language_model.vllm_model import VLLM
from tests.conftest import is_vllm_enabled


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
    llm = VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
        },
        tokenizer_kwargs={"use_fast": False},
        custom_chat_template=custom_chat_template,
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


# With 1 1 1, the continuation was not 1.
TEST_TEMPLATE = """
{%- if fill_with_zeros is defined and fill_with_zeros is true -%}
0 0 0 0 0 0 0 0 0 0 0
{%- else -%}
x x x x x x x x x x x
{%- endif -%}
"""


@pytest.fixture(scope="module")
def chat_lm_with_fill_zeros() -> VLLM:
    """
    VLLM instance with enable_thinking=True in apply_chat_template_kwargs.
    """
    llm = VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
        },
        tokenizer_kwargs={"use_fast": False},
        custom_chat_template=TEST_TEMPLATE,
        apply_chat_template_kwargs={"fill_with_zeros": True},
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def chat_lm_with_fill_xs() -> VLLM:
    """
    VLLM instance with enable_thinking=False in apply_chat_template_kwargs.
    """
    llm = VLLM(
        model="sbintuitions/tiny-lm-chat",
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
        },
        tokenizer_kwargs={"use_fast": False},
        custom_chat_template=TEST_TEMPLATE,
        apply_chat_template_kwargs={"fill_with_zeros": False},
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_custom_chat_template_is_given(chat_lm_with_custom_chat_template: VLLM) -> None:
    responses = chat_lm_with_custom_chat_template.generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=10,
    )
    assert len(responses) == 1
    assert responses[0].text.strip().startswith("0 0")


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_apply_chat_template_kwargs_is_used_with_fill_zeros(chat_lm_with_fill_zeros: VLLM) -> None:
    responses = chat_lm_with_fill_zeros.generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=10,
    )
    assert len(responses) == 1
    assert responses[0].text.strip().startswith("0 0")


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_apply_chat_template_kwargs_is_used_with_fill_ones(chat_lm_with_fill_xs: VLLM) -> None:
    responses = chat_lm_with_fill_xs.generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=10,
    )
    assert len(responses) == 1
    assert responses[0].text.strip().startswith("x x")
