from __future__ import annotations

import re

from .base import StringProcessor


class SimpleEvalMGSMProcessor(StringProcessor):
    """
    StringProcessor that extracts and normalize numerical expression following
    https://github.com/openai/simple-evals/blob/main/mgsm_eval.py.
    This processor is slightly different from that of simple-evals' in that it includes
    an option that does not require the response to start with a specified string.
    This is because when model is not trained on instruction following dataset, it should
    be hard to follow instruction to starts with specified string.

    Examples:
        >>> from flexeval.core.string_processor import SimpleEvalMGSMProcessor
        >>> processor = SimpleEvalMGSMProcessor()
        >>> text = "Step 1: 30.0 + 20.0 = 50.0\\nStep 2: 50.0 × 40.0 = 2,000.0\\nAnswer: 2,000.0"
        >>> print(processor(text))
        2000
    """

    def __init__(self, answer_prefix: str | None = None) -> None:
        self.answer_prefix = answer_prefix

    def __call__(self, text: str) -> str:
        if self.answer_prefix is not None:
            text = text.split(self.answer_prefix)[-1].strip()

        # find all the numbers (including decimals) in the string
        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))

        prediction = numbers[-1].rstrip(".") if numbers else ""

        if "." in prediction:
            prediction = prediction.rstrip("0").rstrip(".")
        return prediction.replace(",", "")


class RemoveCommaProcessor(StringProcessor):
    """
    StringProcessor that remove comma following https://github.com/openai/simple-evals/blob/main/mgsm_eval.py.

    Examples:
        >>> from flexeval.core.string_processor import RemoveCommaProcessor
        >>> processor = RemoveCommaProcessor()
        >>> text = "3,000"
        >>> print(processor(text))
        3000
    """

    def __init__(self) -> None:
        pass

    def __call__(self, text: str) -> str:
        return text.replace(",", "")
