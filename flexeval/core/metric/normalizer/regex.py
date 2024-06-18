import re

from .base import Normalizer


class RegexNormalizer(Normalizer):
    """
    Normalizer that extracts the last match of a regex pattern.

    Args:
        pattern: The regex pattern to extract.
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern, flags=re.DOTALL)

    def normalize(self, text: str) -> str:
        found = self._pattern.findall(text)
        if not found:
            return ""
        return found[-1]
