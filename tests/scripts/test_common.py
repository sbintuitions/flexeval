import time

import pytest

from flexeval.scripts.common import Timer, get_env_metadata


def test_get_env_metadata() -> None:
    metadata = get_env_metadata()
    if metadata["git_hash"] is not None:
        assert isinstance(metadata["git_hash"], str)
    assert isinstance(metadata["python_version"], str)
    assert isinstance(metadata["flexeval_version"], str)
    assert isinstance(metadata["installed_packages"], list)


def test_timer() -> None:
    with Timer() as timer:
        time.sleep(1)
    assert timer.time == pytest.approx(1, abs=0.1)
