from flexeval import TransformersTokenizer


def test_transformers_tokenizer() -> None:
    tokenizer = TransformersTokenizer(path="sbintuitions/tiny-lm")
    assert tokenizer.tokenize("Hello world!") == ["▁", "Hello", "▁world", "!"]
