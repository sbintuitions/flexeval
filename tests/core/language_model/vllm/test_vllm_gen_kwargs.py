import pytest

from flexeval.core.language_model.vllm_model import VLLM, LanguageModel
from tests.conftest import is_vllm_enabled


@pytest.fixture(scope="module")
def lm_with_default_kwargs() -> VLLM:
    llm = VLLM(
        model="sbintuitions/tiny-lm",
        default_gen_kwargs={"max_new_tokens": 1},
        model_kwargs={
            "seed": 42,
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
        },
    )
    yield llm
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not is_vllm_enabled(), reason="vllm library is not installed")
def test_if_gen_kwargs_work_as_expected(lm_with_default_kwargs: LanguageModel) -> None:
    # check if the default gen_kwargs is used and the max_new_tokens is 1
    text = lm_with_default_kwargs.complete_text("000000")
    assert len(text) == 1

    # check if the gen_kwargs will be overwritten by the given gen_kwargs
    text = lm_with_default_kwargs.complete_text("000000", max_new_tokens=10)
    assert len(text) > 1
