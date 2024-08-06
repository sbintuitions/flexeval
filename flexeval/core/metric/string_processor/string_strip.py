from .base import StringProcessor


class StringStrip(StringProcessor):
    """Strip leading and trailing whitespaces from a string.

    Examples:
        >>> from flexeval import StringStrip
        >>> processor = StringStrip()
        >>> text = " ABC"
        >>> normalized_text = processor(text)
        >>> print(normalized_text)
        ABC
    """

    def __call__(self, text: str) -> str:
        return text.strip()
