import unicodedata

from .base import StringProcessor


class NFKCNormalizer(StringProcessor):
    """This processor returns a NFKC normalized string.

    Examples:
        >>> from flexeval import NFKCNormalizer
        >>> processor = NFKCNormalizer()
        >>> text = "０１２３ＡＢＣ"
        >>> normalized_text = processor(text)
        >>> print(normalized_text)
        0123ABC
    """

    def __call__(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)
