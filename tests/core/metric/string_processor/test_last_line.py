import pytest

from flexeval.core.metric.string_processor import LastLineExtractor


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("answer", "answer"),
        ("evidence\nanswer", "answer"),
        ("0\n1\n2", "2"),
        ("", ""),
    ],
)
def test_last_line_extractor(before: str, after: str) -> None:
    extractor = LastLineExtractor()
    assert extractor(before) == after
