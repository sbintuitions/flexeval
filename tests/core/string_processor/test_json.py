import pytest

from flexeval.core.string_processor import JsonNormalizer


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("", "{}"),  # 空文字列 → 空JSON
        ("not a json", "{}"),  # 無効なJSON → 空JSON
        ('{"b": 1, "a": 2}', '{"a": 2, "b": 1}'),  # ソートされる
        ("[1, 2, 3]", "[1, 2, 3]"),  # 配列も処理できる
        ("42", "42"),  # 数値も処理できる
        ('"hello"', '"hello"'),  # 文字列型のJSONも処理できる
    ],
)
def test_aio_processor(before: str, after: str) -> None:
    processor = JsonNormalizer()
    assert processor(before) == after
