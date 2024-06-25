from .base import StringProcessor


class NoopNormalizer(StringProcessor):
    r"""
    A normalizer that does nothing.
    Some metrics apply normalization to both the LM outputs and references by default.
    If you want to explicitly disable normalization for the references, you can use this normalizer.

    Examples:
        >>> from flexeval import ExactMatch, NoopNormalizer, RegexNormalizer
        >>> metric = ExactMatch(normalizer=RegexNormalizer(r"\d+"), reference_normalizer=NoopNormalizer())
        >>> lm_output = "The answer is 10."
        >>> reference = "10"
        >>> print(metric.evaluate([lm_output], [[reference]]))

    """

    def __call__(self, text: str) -> str:
        return text
