from __future__ import annotations

import sacrebleu

from .base import Metric, MetricResult


class BLEU(Metric):
    """An implementation of [BLEU](https://aclanthology.org/P02-1040/).
    The calculation is based on the [sacrebleu](https://github.com/mjpost/sacrebleu) library.

    Args:
        tokenize_option: Tokenization option for sacrebleu.
            If `None`, sacrebleu will use the default tokenization.
    """

    def __init__(self, tokenize_option: str | None = None) -> None:
        self._bleu = sacrebleu.metrics.BLEU(tokenize=tokenize_option)

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

        # we need restructure the references to match the format expected by sacrebleu
        max_num_refs = max(len(refs) for refs in references_list)
        references_for_sacrebleu: list[list[str]] = []
        for i in range(max_num_refs):
            set_of_references: list[str] = []
            for refs_for_source in references_list:
                if i < len(refs_for_source):
                    set_of_references.append(refs_for_source[i])
                else:
                    set_of_references.append("")
            references_for_sacrebleu.append(set_of_references)

        bleu = self._bleu.corpus_score([o.strip() for o in lm_outputs], references_for_sacrebleu)
        sentence_bleu_list = [
            self._bleu.sentence_score(o.strip(), refs) for o, refs in zip(lm_outputs, references_list)
        ]

        return MetricResult(
            {
                "bleu_score": bleu.score / 100,
                "bleu_bp": bleu.bp,
                "bleu_signature": self._bleu.get_signature(),
            },
            instance_details=[{"bleu_score": b.score / 100, "bleu_bp": b.bp} for b in sentence_bleu_list],
        )
