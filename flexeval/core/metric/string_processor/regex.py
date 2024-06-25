import re

from .base import StringProcessor


class RegexExtractor(StringProcessor):
    """
    StringProcessor that extracts the last match of a regex pattern.
    Useful to extract an answer after a step-by-step derivation.

    Args:
        pattern: The regex pattern to extract.

    Examples:
        >>> from flexeval import RegexExtractor
        >>> processor = RegexExtractor(r"Answer: (.*)")
        >>> text = "Step 1: 3 + 2 = 5\\nStep 2: 5 × 4 = 20\\nAnswer: 20"
        >>> print(processor(text))
        20
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern, flags=re.DOTALL)

    def __call__(self, text: str) -> str:
        found = self._pattern.findall(text)
        if not found:
            return ""
        return found[-1]
