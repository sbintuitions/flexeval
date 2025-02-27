import pytest

from flexeval.core.string_processor import StringLower


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("ABCDefg", "abcdefg"),
    ],
)
def test_aio_processor(before: str, after: str) -> None:
    processor = StringLower()
    assert processor(before) == after
