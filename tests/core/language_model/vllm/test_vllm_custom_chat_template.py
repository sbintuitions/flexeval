import pytest
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

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
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_custom_chat_template_is_given(chat_lm_with_custom_chat_template: VLLM) -> None:
    responses = chat_lm_with_custom_chat_template.batch_generate_chat_response(
        [[{"role": "user", "content": "こんにちは。"}]],
        max_new_tokens=10,
    )
    assert len(responses) == 1
    assert responses[0].strip().startswith("0 0")
