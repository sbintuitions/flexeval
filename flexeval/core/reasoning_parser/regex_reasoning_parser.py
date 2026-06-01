import re

from flexeval.core.reasoning_parser.base import Reasoning, ReasoningParser


class UnifiedRegexReasoningParser(ReasoningParser):
    """Extracts reasoning and content using a single regex with named groups.

    The pattern must contain named groups ``(?P<reasoning_content>...)`` and/or
    ``(?P<content>...)`` to populate the corresponding fields of :class:`Reasoning`.
    """

    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern

    def __call__(self, raw_text: str) -> Reasoning:
        m = re.search(self.pattern, raw_text, re.DOTALL)
        if m is None:
            return Reasoning(text=None, reasoning_text=None)
        gd = m.groupdict()
        return Reasoning(
            text=gd.get("content"),
            reasoning_text=gd.get("reasoning_content"),
        )


class SeparatedRegexReasoningParser(ReasoningParser):
    """Extracts reasoning and content using two independent regex patterns.

    Each pattern is applied separately.  If a pattern has a capture group,
    the last group is used; otherwise the full match is used.
    """

    def __init__(
        self,
        pattern_for_content: str,
        pattern_for_reasoning_content: str,
    ) -> None:
        super().__init__()
        self.pattern_for_content = re.compile(pattern_for_content, re.DOTALL)
        self.pattern_for_reasoning_content = re.compile(pattern_for_reasoning_content, re.DOTALL)

    def __call__(self, raw_text: str) -> Reasoning:
        content = None
        reasoning_text = None

        found_content = self.pattern_for_content.findall(raw_text)
        if found_content:
            content = found_content[-1]

        found_reasoning = self.pattern_for_reasoning_content.findall(raw_text)
        if found_reasoning:
            reasoning_text = found_reasoning[-1]

        return Reasoning(text=content, reasoning_text=reasoning_text)
