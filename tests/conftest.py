from typing import Any, Generator

import pytest
from loguru import logger


def is_vllm_enabled() -> bool:
    try:
        import torch
        import vllm  # noqa: F401

        return torch.cuda.device_count() > 0
    except ImportError:
        return False


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture) -> Generator[pytest.LogCaptureFixture, Any, None]:
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)
