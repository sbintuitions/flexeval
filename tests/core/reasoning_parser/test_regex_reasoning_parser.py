import pytest

from flexeval.core.reasoning_parser import Reasoning, SeparatedRegexReasoningParser, UnifiedRegexReasoningParser


@pytest.mark.parametrize(
    ("pattern", "raw_text", "expected"),
    [
        (
            r"<think>(?P<reasoning_content>.*?)</think>(?P<content>.*)",
            "<think>step by step</think>final answer",
            Reasoning(text="final answer", reasoning_text="step by step"),
        ),
        (
            r"<think>(?P<reasoning_content>.*?)</think>(?P<content>.*)",
            "<think>reasoning</think>",
            Reasoning(text="", reasoning_text="reasoning"),
        ),
        (
            r"<think>(?P<reasoning_content>.*?)</think>(?P<content>.*)",
            "no tags here",
            Reasoning(text=None, reasoning_text=None),
        ),
        (
            r"(?P<reasoning_content>.*)",
            "only reasoning",
            Reasoning(text=None, reasoning_text="only reasoning"),
        ),
        (
            r"(?P<content>.*)",
            "only content",
            Reasoning(text="only content", reasoning_text=None),
        ),
        (
            r"<think>(?P<reasoning_content>.*?)</think>(?P<content>.*)",
            "<think>line1\nline2</think>answer",
            Reasoning(text="answer", reasoning_text="line1\nline2"),
        ),
    ],
)
def test_unified_regex_reasoning_parser(pattern: str, raw_text: str, expected: Reasoning) -> None:
    parser = UnifiedRegexReasoningParser(pattern=pattern)
    assert parser(raw_text) == expected


@pytest.mark.parametrize(
    ("pattern_for_content", "pattern_for_reasoning_content", "raw_text", "expected"),
    [
        (
            r"<answer>(.*?)</answer>",
            r"<think>(.*?)</think>",
            "<think>step by step</think><answer>final answer</answer>",
            Reasoning(text="final answer", reasoning_text="step by step"),
        ),
        (
            r"<answer>(.*?)</answer>",
            r"<think>(.*?)</think>",
            "<think>reasoning</think>",
            Reasoning(text=None, reasoning_text="reasoning"),
        ),
        (
            r"<answer>(.*?)</answer>",
            r"<think>(.*?)</think>",
            "<answer>answer</answer>",
            Reasoning(text="answer", reasoning_text=None),
        ),
        (
            r"<answer>(.*?)</answer>",
            r"<think>(.*?)</think>",
            "no tags",
            Reasoning(text=None, reasoning_text=None),
        ),
        (
            r"<answer>(.*?)</answer>",
            r"<think>(.*?)</think>",
            "<think>r1</think><think>r2</think><answer>a1</answer><answer>a2</answer>",
            Reasoning(text="a2", reasoning_text="r2"),
        ),
        (
            r"<answer>.*?</answer>",
            r"<think>.*?</think>",
            "<think>reasoning</think><answer>answer</answer>",
            Reasoning(text="<answer>answer</answer>", reasoning_text="<think>reasoning</think>"),
        ),
        (
            r"<answer>(.*?)</answer>",
            r"<think>(.*?)</think>",
            "<think>line1\nline2</think><answer>result</answer>",
            Reasoning(text="result", reasoning_text="line1\nline2"),
        ),
    ],
)
def test_separated_regex_reasoning_parser(
    pattern_for_content: str,
    pattern_for_reasoning_content: str,
    raw_text: str,
    expected: Reasoning,
) -> None:
    parser = SeparatedRegexReasoningParser(
        pattern_for_content=pattern_for_content,
        pattern_for_reasoning_content=pattern_for_reasoning_content,
    )
    assert parser(raw_text) == expected
