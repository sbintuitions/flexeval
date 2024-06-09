from flexeval.core.metric.tokenizer import WhitespaceTokenizer


def test_whitespace_tokenizer() -> None:
    tokenizer = WhitespaceTokenizer()
    assert tokenizer.tokenize("hello world") == ["hello", "world"]
