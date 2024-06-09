from flexeval.core.metric.tokenizer import SacreBleuTokenizer


def test_sacrebleu_zh_tokenizer() -> None:
    tokenizer = SacreBleuTokenizer(name="zh")
    assert tokenizer.tokenize("教我中文吧") == ["教", "我", "中", "文", "吧"]


def test_sacrebleu_13a_tokenizer() -> None:
    tokenizer = SacreBleuTokenizer(name="13a")
    assert tokenizer.tokenize("I'm a student.") == ["I'm", "a", "student", "."]
