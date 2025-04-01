import functools
from typing import Callable

import pytest

from flexeval.core.language_model.vllm_model import (
    VLLM,
)
from tests.conftest import is_vllm_enabled
from tests.core.language_model.base import BaseLanguageModelTest


@pytest.fixture(scope="module")
def lm_init_func(model: str = "sbintuitions/tiny-lm") -> Callable[..., VLLM]:
    # use float32 because half precision is not supported in some hardware
    return functools.partial(
        VLLM,
        model=model,
        tokenizer_kwargs={"use_fast": False},
        default_gen_kwargs={"temperature": 0.0},
    )


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
class TestVLLM(BaseLanguageModelTest):
    @pytest.fixture()
    def lm(self, chat_lm: VLLM) -> VLLM:
        return chat_lm

    @pytest.fixture()
    def chat_lm(self, chat_lm: VLLM) -> VLLM:
        return chat_lm
