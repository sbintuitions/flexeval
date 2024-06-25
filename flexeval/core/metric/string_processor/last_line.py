from .base import StringProcessor


class LastLineExtractor(StringProcessor):
    """Extract the last line from a string.

    Examples:
        >>> from flexeval import LastLineExtractor
        >>> normalizer = LastLineExtractor()
        >>> text = "Answer\\nFUJI-YAMA"
        >>> print(normalizer(text))
        FUJI-YAMA
    """

    def __call__(self, text: str) -> str:
        return text.split("\n")[-1]
