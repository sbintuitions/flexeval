from __future__ import annotations

import pytest

from flexeval.core.language_model.base import LMOutput


@pytest.fixture(params=["as-str", "as-LMOutput"], ids=["as-str", "as-LMOutput"])
def as_lm_output(request: pytest.FixtureRequest) -> str:
    """
    Fixture that parameterizes tests to run with both string and LMOutput formats.

    This fixture returns either "as-str" or "as-LMOutput" to control how the
    lm_outputs fixture formats test data. Tests using this fixture will run
    twice - once with plain strings and once with LMOutput objects.

    Returns:
        str: Either "as-str" or "as-LMOutput" indicating the desired format.
    """
    return request.param


@pytest.fixture
def lm_outputs(request: pytest.FixtureRequest, as_lm_output: str) -> list[str] | list[LMOutput]:
    """
    Fixture that converts parameterized string lists to either strings or LMOutput objects.

    This fixture works with @pytest.mark.parametrize to accept a list of strings
    and convert them to the appropriate format based on the as_lm_output fixture.
    It enables testing metric classes with both input formats to ensure they
    handle both str and LMOutput inputs correctly.

    Args:
        request: pytest fixture request containing the parameterized data.
        as_lm_output: Format indicator from as_lm_output fixture.

    Returns:
        list[str] | list[LMOutput]: The test data formatted as strings or LMOutput objects.

    Example:
        @pytest.mark.parametrize("lm_outputs", [["hello", "world"]], indirect=True)
        def test_metric(lm_outputs):
            # lm_outputs will be either ["hello", "world"] or [LMOutput(text="hello"), LMOutput(text="world")]
            pass
    """
    raw: list[str] = request.param
    if as_lm_output == "as-LMOutput":
        return [LMOutput(text=x) for x in raw]
    return raw
