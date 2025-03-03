from flexeval import TiktokenTokenizer


def test_tokenizer_from_tokenizer_name() -> None:
    tokenizer = TiktokenTokenizer(tokenizer_name="o200k_base")
    assert tokenizer.tokenize("Hello world!") == ["Hello", " world", "!"]


def test_tokenizer_from_model_name() -> None:
    tokenizer = TiktokenTokenizer(model_name="gpt-4o")
    assert tokenizer.tokenize("Hello world!") == ["Hello", " world", "!"]
