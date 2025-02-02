from .base import StringProcessor


class StringLower(StringProcessor):
    """This processor returns a lowercased string.

    Examples:
        >>> from flexeval import StringLower
        >>> processor = StringLower()
        >>> text = "ABCDefg"
        >>> normalized_text = processor(text)
        >>> print(normalized_text)
        abcdefg
    """

    def __call__(self, text: str) -> str:
        return text.lower()
