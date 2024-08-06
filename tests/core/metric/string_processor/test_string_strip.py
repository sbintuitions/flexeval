import pytest

from flexeval.core.metric.string_processor import StringStrip


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (" answer", "answer"),
        ("answer", "answer"),
        ("\n1", "1"),
        ("", ""),
    ],
)
def test_last_line_extractor(before: str, after: str) -> None:
    extractor = StringStrip()
    assert extractor(before) == after
