from __future__ import annotations

import functools

import sacrebleu

from flexeval.core.metric.utils import aggregate_category_wise_scores
from flexeval.core.string_processor.base import StringProcessor

from .base import Metric, MetricResult


class BLEU(Metric):
    """An implementation of [BLEU](https://aclanthology.org/P02-1040/).
    The calculation is based on the [sacrebleu](https://github.com/mjpost/sacrebleu) library.

    Args:
        tokenize_option: Tokenization option for sacrebleu.
            If `None`, sacrebleu will use the default tokenization.
        lm_output_processor:
            StringProcessor or a list of StringProcessor to be applied to the model outputs before comparison.
        reference_processor: StringProcessor or list of StringProcessor to apply to the references before comparison.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in extra_info.

    Examples:
        >>> from flexeval import BLEU
        >>> bleu = BLEU()
        >>> lm_outputs = ["I am a student .", "I am a teacher ."]
        >>> references_list = [["I am a student .", "I am a learner ."], ["I am a teacher ."]]
        >>> result = bleu.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={
                'bleu_score': 1.0,
                'bleu_bp': 1.0,
                'bleu_signature': nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.1},
                instance_details=[
                    {'bleu_score': 1.0, 'bleu_bp': 1.0},
                    {'bleu_score': 1.0, 'bleu_bp': 1.0}
                ]
            )
    """

    def __init__(
        self,
        tokenize_option: str | None = None,
        lm_output_processor: StringProcessor | list[StringProcessor] | None = None,
        reference_processor: StringProcessor | list[StringProcessor] | None = None,
        category_key: str | None = None,
    ) -> None:
        self._corpus_bleu = sacrebleu.metrics.BLEU(tokenize=tokenize_option)
        # For sentence BLEU, we need to set `effective_order=True` as recommended by sacrebleu.
        self._sentence_bleu = sacrebleu.metrics.BLEU(tokenize=tokenize_option, effective_order=True)

        if isinstance(lm_output_processor, StringProcessor):
            lm_output_processor = [lm_output_processor]
        if isinstance(reference_processor, StringProcessor):
            reference_processor = [reference_processor]

        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor
        self.category_key = category_key

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if len(lm_outputs) != len(references_list):
            msg = (
                f"lm_outputs and references_list must have the same length, "
                f"but got {len(lm_outputs)} and {len(references_list)}."
            )
            raise ValueError(msg)

        if self.lm_output_processors:
            lm_outputs = [
                functools.reduce(lambda x, norm: norm(x), self.lm_output_processors, output) for output in lm_outputs
            ]

        if self.reference_processors:
            references_list = [
                [functools.reduce(lambda x, norm: norm(x), self.reference_processors, ref) for ref in references]
                for references in references_list
            ]

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

        bleu = self._corpus_bleu.corpus_score([o.strip() for o in lm_outputs], references_for_sacrebleu)
        sentence_bleu_list = [
            self._sentence_bleu.sentence_score(o.strip(), refs) for o, refs in zip(lm_outputs, references_list)
        ]

        summary = {
            "bleu_score": bleu.score / 100,
            "bleu_bp": bleu.bp,
            "bleu_signature": self._corpus_bleu.get_signature(),
        }

        if self.category_key:
            categories = [extra_info[self.category_key] for extra_info in extra_info_list]
            sentence_bleu_score_list = [b.score / 100 for b in sentence_bleu_list]
            category_wise_scores = aggregate_category_wise_scores(sentence_bleu_score_list, categories)
            for category, category_wise_score in category_wise_scores.items():
                summary[f"sentence_bleu_score/{category}"] = category_wise_score

        return MetricResult(
            summary,
            instance_details=[{"bleu_score": b.score / 100, "bleu_bp": b.bp} for b in sentence_bleu_list],
        )
