import pytest

from flexeval.core.string_processor import AIONormalizer


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("さまぁ～ず", "さまぁ〜ず"),  # 波線の正規化
        ("３回", "3回"),  # NFKC正規化
        ("Apple", "apple"),  # upper case → lower case 置換
        ("「富士山」", "富士山"),  # 鉤括弧を削除
        ("『富士山』", "富士山"),  # 鉤括弧を削除
        ("レオナルド・ダヴィンチ", "レオナルドダヴィンチ"),  # 記号の削除
        ("レオナルド=ダヴィンチ", "レオナルドダヴィンチ"),  # 記号の削除
        ("レオナルド-ダヴィンチ", "レオナルドダヴィンチ"),  # 記号の削除
        ("hyper text markup    language", "hypertextmarkuplanguage"),  # 空白の削除
        ("蛹化(ようか)", "蛹化"),  # 括弧の削除
        ("搦手(からめて)門", "搦手門"),  # 括弧の削除
    ],
)
def test_aio_processor(before: str, after: str) -> None:
    processor = AIONormalizer()
    assert processor(before) == after
