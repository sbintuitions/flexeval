from __future__ import annotations

from flexeval.core.language_model import LMOutput
from flexeval.core.metric.utils import apply_string_processors, extract_text_from_outputs, validate_inputs
from flexeval.core.string_processor.base import StringProcessor
from flexeval.core.tokenizer import Tokenizer, TransformersTokenizer

from .base import Metric, MetricResult


class RepetitionN(Metric):
    """
    A metric that calculates the n-gram repetition ratio in the model's output.
    The repetition n-gram score is defined as:
    $\text{Repetition-N} = 1 - \frac{\text{#Unique n-grams}}{\text{#Total n-grams}}$

    Reference: https://aclanthology.org/2020.acl-main.428

    Args:
        n: The 'n' in n-grams.
        tokenizer: An instance of `flexeval.core.tokenizer.Tokenizer` to tokenize the input and output strings.
    """

    def __init__(
        self,
        n: int = 4,
        tokenizer: Tokenizer | None = None,
        lm_output_processor: StringProcessor | list[StringProcessor] | None = None,
    ) -> None:
        self.n = n
        self.lm_output_processors = lm_output_processor
        self.tokenizer = tokenizer \
            if isinstance(tokenizer, Tokenizer) \
            else TransformersTokenizer("sbintuitions/sarashina2.2-3b-instruct-v0.1")

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)

        # Extract text from LMOutput objects
        lm_outputs = extract_text_from_outputs(lm_outputs)

        # Normalize text data
        lm_outputs = [apply_string_processors(output, self.lm_output_processors) for output in lm_outputs]
        scores = []
        for output in lm_outputs:
            tokens = self.tokenizer.tokenize(output)
            ngrams = [tuple(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)]
            if len(ngrams) == 0:
                scores.append(0.0)
                continue
            unique_ngrams = set(ngrams)
            repetition_n = 1.0 - (len(unique_ngrams) / len(ngrams))
            scores.append(repetition_n)
        average_score = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            summary={f"repetition_{self.n}": average_score},
            instance_details=[{f"repetition_{self.n}": score} for score in scores]
        )
