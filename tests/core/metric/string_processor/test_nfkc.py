import pytest

from flexeval.core.metric.string_processor import NFKCNormalizer


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("０１２３ＡＢＣ", "0123ABC"),
    ],
)
def test_aio_processor(before: str, after: str) -> None:
    processor = NFKCNormalizer()
    assert processor(before) == after
