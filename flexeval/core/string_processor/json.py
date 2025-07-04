import json
from typing import Any

from .base import StringProcessor


class JsonNormalizer(StringProcessor):
    """Parse input text as json, sort by keys, and finally dump into string.
    If input text cannot be parsed, always return '{}'.

    Examples:
        >>> from flexeval import JsonNormalizer
        >>> processor = JsonNormalizer()
        >>> text = '{"b": 1, "a": 2}'
        >>> normalized_text = processor(text)
        >>> print(normalized_text)
        {"a": 2, "b": 1}
    """

    def __call__(self, text: str) -> str:
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        return json.dumps(data, sort_keys=True, ensure_ascii=False)
