import pytest

from flexeval.core.metric.string_processor import RegexExtractor


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("<think>reasoning</think>answer", "answer"),
        ("<think>\nOkay, so let's think...\nThis question is...\n...\n</think>\nanswer", "answer"),
        ("<think>Okay, so let's think...\nThis question is...\n...</think>answer", "answer"),
        ("Okay, so let's think...\nThis question is...\n...\n</think>\nanswer", "answer"),
        ("answer", "answer"),
        ("", ""),
    ],
)
def test_reasoning_tag_extractor(before: str, after: str) -> None:
    extractor = RegexExtractor(r"^(?:.*</think>\s*)?(.*)$")
    assert extractor(before) == after
