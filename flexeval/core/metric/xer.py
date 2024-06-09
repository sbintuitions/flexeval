from __future__ import annotations

from jiwer import cer, wer

from .base import Metric, MetricResult
from .tokenizer import Tokenizer


class XER(Metric):
    """
    Calculate the Character Error Rate (CER) and Word Error Rate (WER) between the model outputs and the references.
    The calculation is based on the [jiwer](https://github.com/jitsi/jiwer) library.

    Args:
        tokenizer: An instance of `Tokenizer` to tokenize the input and output strings.
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if len(lm_outputs) != len(references_list):
            msg = (
                f"lm_outputs and references_list must have the same length, "
                f"but got {len(lm_outputs)} and {len(references_list)}."
            )
            raise ValueError(msg)

        # we only need the first reference
        references = [references[0] for references in references_list]

        if self.tokenizer:
            tokenized_lm_outputs = [" ".join(self.tokenizer.tokenize(lm_output)) for lm_output in lm_outputs]
            tokenized_references = [" ".join(self.tokenizer.tokenize(reference)) for reference in references]
        else:
            tokenized_lm_outputs = lm_outputs
            tokenized_references = references

        cer_score = cer(references, lm_outputs)
        wer_score = wer(tokenized_references, tokenized_lm_outputs)

        return MetricResult(
            {
                "cer_score": cer_score,
                "wer_score": wer_score,
            },
        )
