from .base import StringProcessor


class LastLineExtractor(StringProcessor):
    """Extract the last line from a string.

    Examples:
        >>> from flexeval import LastLineExtractor
        >>> processor = LastLineExtractor()
        >>> text = "Answer\\nFUJI-YAMA"
        >>> print(processor(text))
        FUJI-YAMA
    """

    def __call__(self, text: str) -> str:
        return text.split("\n")[-1]
