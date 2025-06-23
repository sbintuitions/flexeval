from __future__ import annotations

from typing import Any

import pytest

from flexeval.core.utils.chat_util import find_first_turn_for_response


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([{"role": "user", "content": "hello"}], 0),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "hello"},
            ],
            1,
        ),
        (
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "who are you?"},
            ],
            1,
        ),
        (
            [
                {"role": "developer", "content": "developer message"},
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "who are you?"},
            ],
            1,
        ),
    ],
)
def test_find_first_turn_for_response(messages: list[dict[str, Any]], expected: int) -> None:
    assert find_first_turn_for_response(messages) == expected
