from __future__ import annotations

from rouge import Rouge as RougeCalculator

from flexeval.core.language_model.base import LMOutput
from flexeval.core.tokenizer import Tokenizer

from .base import Metric, MetricResult
from .utils import extract_text_from_outputs, validate_inputs


class ROUGE(Metric):
    """An implementation of [ROUGE](https://aclanthology.org/W04-1013/).

    The calculation is based on the [rouge](https://github.com/pltrdy/rouge) library.

    Args:
        tokenizer: An instance of `Tokenizer` to tokenize the input and output strings.
        max_output_tokens: Maximum number of tokens to consider from model outputs.
            If specified, only model outputs will be truncated to this length.

    Examples:
        >>> from flexeval import ROUGE
        >>> from flexeval import WhitespaceTokenizer
        >>> tokenizer = WhitespaceTokenizer()
        >>> rouge = ROUGE(tokenizer)
        >>> lm_outputs = ["I am a student .", "I am a teacher ."]
        >>> references_list = [["I am a student .", "I am a learner ."], ["I am a teacher ."]]
        >>> result = rouge.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={'rouge1': 0.999999995, 'rouge2': 0.999999995, 'rougeL': 0.999999995},
            instance_details=[
                {'rouge1': 0.999999995, 'rouge2': 0.999999995, 'rougeL': 0.999999995},
                {'rouge1': 0.999999995, 'rouge2': 0.999999995, 'rougeL': 0.999999995}
            ]
        )
    """

    def __init__(self, tokenizer: Tokenizer, max_output_tokens: int | None = None) -> None:
        self._tokenizer = tokenizer
        self._max_output_tokens = max_output_tokens

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)

        # Extract text from LMOutput objects
        lm_outputs = extract_text_from_outputs(lm_outputs)

        # Normalize text data - we only need the first reference
        target_summaries = [references[0] for references in references_list]

        tokenized_lm_outputs = [" ".join(self._tokenizer.tokenize(lm_output)) for lm_output in lm_outputs]
        tokenized_target_summaries = [
            " ".join(self._tokenizer.tokenize(target_summary)) for target_summary in target_summaries
        ]

        # Truncate model outputs if max_output_tokens is specified
        if self._max_output_tokens is not None:
            tokenized_lm_outputs = [
                " ".join(output.split()[: self._max_output_tokens]) for output in tokenized_lm_outputs
            ]

        # replace empty string with " " to avoid "ValueError: Hypothesis is empty" from rouge
        tokenized_lm_outputs = [o if o else " " for o in tokenized_lm_outputs]

        # Compute metrics
        rouge = RougeCalculator()
        score_outputs = rouge.get_scores(
            tokenized_lm_outputs,
            tokenized_target_summaries,
        )

        rouge1_list = [o["rouge-1"]["f"] for o in score_outputs]
        rouge2_list = [o["rouge-2"]["f"] for o in score_outputs]
        rouge_l_list = [o["rouge-l"]["f"] for o in score_outputs]

        # we only need the f1 score
        return MetricResult(
            {
                "rouge1": sum(rouge1_list) / len(rouge1_list),
                "rouge2": sum(rouge2_list) / len(rouge2_list),
                "rougeL": sum(rouge_l_list) / len(rouge_l_list),
            },
            instance_details=[
                {"rouge1": r1, "rouge2": r2, "rougeL": rL} for r1, r2, rL in zip(rouge1_list, rouge2_list, rouge_l_list)
            ],
        )
