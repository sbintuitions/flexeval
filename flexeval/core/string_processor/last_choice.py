import re
import string
from typing import Literal

from .base import StringProcessor


class LastChoiceExtractor(StringProcessor):
    """
    StringProcessor that extracts the last mentioned multiple-choice option symbol.
    Useful to extract an answer after a step-by-step derivation.

    MCQ benchmark scores (e.g. MMMU/JMMMU `mcq_exact_match`) depend on this
    extraction logic; changing it silently shifts reported scores, so keep it
    stable across releases.

    Args:
        lang: Language of the response, "en" or "ja".
            With "ja", option symbols surrounded by Japanese characters
            (e.g. "正解はAです。") are also detected.
        option_type: The style of option symbols:
            "number" (1, 2, ...), "alphabet" (a, b, ...), or "Alphabet" (A, B, ...).
        max_num_options: The maximum number of options.

    Examples:
        >>> from flexeval import LastChoiceExtractor
        >>> processor = LastChoiceExtractor(lang="en")
        >>> text = "The answer is A."
        >>> print(processor(text))
        A
    """

    def __init__(
        self,
        lang: Literal["ja", "en"] = "en",
        option_type: Literal["number", "alphabet", "Alphabet"] = "Alphabet",
        max_num_options: int = 4,
    ) -> None:
        self.lang = lang
        if option_type == "number":
            self.choices = [str(i) for i in range(1, max_num_options + 1)]
        elif option_type == "alphabet":
            self.choices = list(string.ascii_lowercase[:max_num_options])
        elif option_type == "Alphabet":
            self.choices = list(string.ascii_uppercase[:max_num_options])
        else:
            msg = f"Unsupported option_type: {option_type}"
            raise ValueError(msg)

    def _remove_punctuations(self, response: str) -> str:
        punctuation = string.punctuation
        if self.lang == "ja":
            # add full-width punctuations
            punctuation += "！”＃＄％＆’（）*+，−．／：；＜＝＞？＠［＼］＾＿｀｛｜｝〜、。￥・"
        return response.strip().strip(punctuation)

    def _find_choice_with_brackets(self, response: str) -> list[str]:
        # e.g. (A), (B), ...
        return [c for c in self.choices if f"({c})" in response]

    def _find_choice_without_brackets(self, response: str) -> list[str]:
        # e.g. A, B, ...
        return [c for c in self.choices if re.search(rf"\b{c}\b", response)]

    def _find_choice_between_japanese(self, response: str) -> list[str]:
        # e.g. 正解はAです。
        japanese_char_pattern = r"[\u3040-\u30FF\u4E00-\u9FFF]"
        return [c for c in self.choices if re.search(rf"{japanese_char_pattern}{c}{japanese_char_pattern}", response)]

    def __call__(self, text: str) -> str:
        candidates = self._find_choice_with_brackets(text)
        if not candidates:
            text = self._remove_punctuations(text)
            candidates = self._find_choice_without_brackets(text)
        if not candidates and self.lang == "ja":
            candidates = self._find_choice_between_japanese(text)

        if not candidates:
            return ""
        # choose the last mentioned option if multiple options are extracted
        return max(candidates, key=text.rfind)
