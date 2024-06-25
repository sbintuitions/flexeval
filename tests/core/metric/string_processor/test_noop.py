import pytest

from flexeval.core.metric.string_processor import NoopNormalizer


@pytest.mark.parametrize(
    "text",
    [
        "answer",
        "evidence\nanswer",
        "",
    ],
)
def test_last_line_extractor(text: str) -> None:
    extractor = NoopNormalizer()
    assert extractor(text) == text
