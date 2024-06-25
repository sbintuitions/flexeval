from .base import StringProcessor


class NoopNormalizer(StringProcessor):
    r"""
    A processor that does nothing.
    Some metrics apply normalization to both the LM outputs and references by default.
    If you want to explicitly disable normalization for the references, you can use this processor.

    Examples:
        >>> from flexeval import ExactMatch, NoopNormalizer, RegexExtractor
        >>> metric = ExactMatch(processor=RegexExtractor(r"\d+"), reference_processor=NoopNormalizer())
        >>> lm_output = "The answer is 10."
        >>> reference = "10"
        >>> print(metric.evaluate([lm_output], [[reference]]))

    """

    def __call__(self, text: str) -> str:
        return text
