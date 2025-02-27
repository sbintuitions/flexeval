import re
import unicodedata

from .base import StringProcessor


class AIONormalizer(StringProcessor):
    """StringProcessor used for AI王 (AI king) question answering task.
    This is adapted from
    [the official script](https://github.com/cl-tohoku/aio4-bpr-baseline/blob/c5a226296b5e1c403268016dc7136147bbb515fe/compute_score.py).

    Examples:
        >>> from flexeval import AIONormalizer
        >>> processor = AIONormalizer()
        >>> text = "「蛹化(ようか)」"
        >>> normalized_text = processor(text)
        >>> print(normalized_text)
        蛹化
    """

    def __call__(self, text: str) -> str:
        # substitute some symbols that will not be replaced by unicode normalization
        text = text.replace("～", "〜")

        # unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # lowercase alphabetical characters
        text = text.lower()

        # remove kagi-kakkos
        text = re.sub(r"「(.*?)」", r"\1", text)
        text = re.sub(r"『(.*?)』", r"\1", text)

        # remove some punctuation marks
        text = text.replace("・", "")
        text = text.replace("=", "")
        text = text.replace("-", "")

        # compress whitespaces
        text = re.sub(r"\s+", "", text).strip()

        # remove parenthesis: 蛹化(ようか)　→　蛹化
        return re.sub(r"\((.*?)\)", "", text)
