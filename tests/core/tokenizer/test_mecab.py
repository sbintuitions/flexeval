from unittest.mock import Mock, patch

import pytest

from flexeval import MecabTokenizer


@pytest.fixture
def mocked_fugashi_tagger() -> Mock:
    # Mock fugashi.Tagger as it requires downloading the MeCab dictionary
    with patch("fugashi.Tagger") as mock_tagger:
        mock_tagger_instance = Mock()
        mock_tagger.return_value = mock_tagger_instance
        mock_token1 = Mock()
        mock_token1.surface = "token1"
        mock_token2 = Mock()
        mock_token2.surface = "token2"
        mock_tagger_instance.return_value = [mock_token1, mock_token2]
        yield mock_tagger


def test_mecab_tokenizer(mocked_fugashi_tagger: Mock) -> None:
    tokenizer = MecabTokenizer()
    text = "これはテストです。"
    tokens = tokenizer.tokenize(text)

    assert tokens == ["token1", "token2"]
    mocked_fugashi_tagger.assert_called_once_with("-Owakati")
    mocked_fugashi_tagger.return_value.assert_called_once_with(text)
