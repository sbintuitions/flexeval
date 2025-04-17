from __future__ import annotations

import functools
from typing import Literal

from flexeval.core.metric.base import Metric, MetricResult
from flexeval.core.metric.utils import aggregate_category_wise_scores
from flexeval.core.string_processor.base import StringProcessor
from flexeval.core.string_processor.lower import StringLower
from flexeval.core.tokenizer.base import Tokenizer
from flexeval.core.tokenizer.sacrebleu_tokenizer import SacreBleuTokenizer


def to_ngram(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


class SARI(Metric):
    """An implementation of SARI, a metric for evaluating text simplification.

    Based on the original implementation [1], but modified to allow configurable settings
    for the maximum n-gram size and tokenizer.
    Additionally, it fixes a bug present in the original implementation [2].
    When used with the default parameters, it produces scores that are
    generally consistent with the [3] Hugging Face evaluate implementation.

    [1] https://github.com/cocoxu/simplification/blob/master/SARI.py
    [2] https://github.com/cocoxu/simplification/issues/6
    [3] https://huggingface.co/spaces/evaluate-metric/sari/blob/main/sari.py

    Args:
        tokenizer: An instance of `Tokenizer` to tokenize the input and output strings.

    Examples:
        >>> from flexeval import SARI
        >>> sari_scorer = SARI(source_key="source")
        >>> lm_outputs = ["About 95 you now get in."]
        >>> references_list = [["About 95 species are currently known.", "About 95 species are now accepted.", "95 species are now accepted."]]
        >>> task_inputs_list = [{"source": "About 95 species are currently accepted."}]
        >>> result = sari_scorer.evaluate(lm_outputs, references_list, task_inputs_list)
        >>> print(result)
        MetricResult(summary={'sari_score': 0.2695360195360195}, instance_details=[{'sari_score': 0.2695360195360195}])
    """  # noqa: E501
    def __init__(
        self,
        source_key: str,
        tokenizer: Tokenizer | Literal["default"] = "default",
        max_ngrams: int = 4,
        category_key: str | None = None,
        source_processor: StringProcessor | list[StringProcessor] | None | Literal["default"] = "default",
        lm_output_processor: StringProcessor | list[StringProcessor] | None | Literal["default"] = "default",
        reference_processor: StringProcessor | list[StringProcessor] | None | Literal["default"] = "default",
    ) -> None:
        if tokenizer == "default":
            tokenizer = SacreBleuTokenizer("13a")
        self._tokenizer = tokenizer
        self.source_key = source_key
        self.max_ngrams = max_ngrams
        self.category_key = category_key
        if source_processor == "default":
            source_processor = StringLower()
        if lm_output_processor == "default":
            lm_output_processor = StringLower()
        if reference_processor == "default":
            reference_processor = StringLower()
        if isinstance(source_processor, StringProcessor):
            source_processor = [source_processor]
        if isinstance(lm_output_processor, StringProcessor):
            lm_output_processor = [lm_output_processor]
        if isinstance(reference_processor, StringProcessor):
            reference_processor = [reference_processor]
        self.source_processors = source_processor
        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor

    def evaluate(self, lm_outputs, references_list, task_inputs_list = None) -> MetricResult:  # noqa: ANN001
        if task_inputs_list is None:
            msg = "SARI requires task_inputs_list"
            raise ValueError(msg)
        sources = [task_input[self.source_key] for task_input in task_inputs_list]

        if not(len(sources) == len(lm_outputs) == len(references_list)):
            msg = (
                f"sources, lm_outputs and references_list must have the same length, "
                f"but got {len(sources)}, {len(lm_outputs)} and {len(references_list)}."
            )
            raise ValueError(msg)

        if self.source_processors:
            sources = [
                functools.reduce(lambda x, norm: norm(x), self.source_processors, src) for src in sources
            ]

        if self.lm_output_processors:
            lm_outputs = [
                functools.reduce(lambda x, norm: norm(x), self.lm_output_processors, output) for output in lm_outputs
            ]

        if self.reference_processors:
            references_list = [
                [functools.reduce(lambda x, norm: norm(x), self.reference_processors, ref) for ref in references]
                for references in references_list
            ]

        sari_list = [
            self._calc_sentence_sari(source, lm_output, references)
            for source, lm_output, references in zip(sources, lm_outputs, references_list)
        ]

        sari_score_list = [s["sari_score"] for s in sari_list]
        sari_add_list = [s["sari_add"] for s in sari_list]
        sari_keep_list = [s["sari_keep"] for s in sari_list]
        sari_del_list = [s["sari_del"] for s in sari_list]

        num_instances = len(sari_list)
        summary = {
            "sari_score": sum(sari_score_list) / num_instances,
            "sari_add": sum(sari_add_list) / num_instances,
            "sari_keep": sum(sari_keep_list) / num_instances,
            "sari_del": sum(sari_del_list) / num_instances,
        }

        if self.category_key:
            categories = [task_input[self.category_key] for task_input in task_inputs_list]
            category_wise_sari_scores = aggregate_category_wise_scores(sari_score_list, categories)
            category_wise_sari_adds = aggregate_category_wise_scores(sari_add_list, categories)
            category_wise_sari_keeps = aggregate_category_wise_scores(sari_keep_list, categories)
            category_wise_sari_dels = aggregate_category_wise_scores(sari_del_list, categories)

            for category, category_wise_score in category_wise_sari_scores.items():
                summary[f"sari_score/{category}"] = category_wise_score
            for category, category_wise_add in category_wise_sari_adds.items():
                summary[f"sari_add/{category}"] = category_wise_add
            for category, category_wise_keep in category_wise_sari_keeps.items():
                summary[f"sari_keep/{category}"] = category_wise_keep
            for category, category_wise_del in category_wise_sari_dels.items():
                summary[f"sari_del/{category}"] = category_wise_del

        return MetricResult(
            summary,
            instance_details=sari_list,
        )


    def _calc_sentence_sari(self, source: str, lm_output: str, references: list[str]) -> dict[str, float]:
        s_words = self._tokenizer.tokenize(source)
        c_words = self._tokenizer.tokenize(lm_output)
        r_words_list = [self._tokenizer.tokenize(reference) for reference in references]

        sari_score = 0.0
        sari_add = 0.0
        sari_keep = 0.0
        sari_del = 0.0
        for n in range(1, self.max_ngrams + 1):  # 1-gram to n-gram
            s_ngrams = to_ngram(s_words, n)
            c_ngrams = to_ngram(c_words, n)
            r_ngrams_list = [to_ngram(r_words, n) for r_words in r_words_list]

            sari_n_score, sari_n_add, sari_n_keep, sari_n_del = self._sari_n(s_ngrams, c_ngrams, r_ngrams_list)
            sari_score += sari_n_score
            sari_add += sari_n_add
            sari_keep += sari_n_keep
            sari_del += sari_n_del

        sari_score /= self.max_ngrams
        sari_add /= self.max_ngrams
        sari_keep /= self.max_ngrams
        sari_del /= self.max_ngrams

        return {"sari_score": sari_score, "sari_add": sari_add, "sari_keep": sari_keep, "sari_del": sari_del}


    def _sari_n(
            self, s_grams: list[str], c_grams: list[str], r_grams_list: list[list[str]]
        ) -> tuple[float, float, float, float]:
        from collections import Counter
        num_ref = len(r_grams_list)
        r_grams_all = [r_gram for r_grams in r_grams_list for r_gram in r_grams]
        r_gram_counter = Counter(r_grams_all)

        s_gram_counter = Counter(s_grams)
        c_gram_counter = Counter(c_grams)

        s_gram_rep = Counter({k: v * num_ref for k, v in s_gram_counter.items()})
        c_gram_rep = Counter({k: v * num_ref for k, v in c_gram_counter.items()})

        # ADD
        add_grams = set(c_gram_counter) - set(s_gram_counter)
        add_good = add_grams & set(r_gram_counter)
        add_all = set(r_gram_counter) - set(s_gram_counter)

        add_prec = len(add_good) / len(add_grams) if add_grams else 0
        add_recall = len(add_good) / len(add_all) if add_all else 0
        add_f1 = (
            2 * add_prec * add_recall / (add_prec + add_recall)
            if (add_prec + add_recall) > 0 else 0
        )

        # KEEP
        keep_rep = s_gram_rep & c_gram_rep
        keep_good = keep_rep & r_gram_counter
        keep_all = s_gram_rep & r_gram_counter

        keep_prec = (
            sum(keep_good[g] / keep_rep[g] for g in keep_good) / len(keep_rep)
            if keep_rep else 0
        )
        keep_recall = (
            sum(keep_good[g] for g in keep_good) / sum(keep_all.values())
            if keep_all else 0
        )
        keep_f1 = (
            2 * keep_prec * keep_recall / (keep_prec + keep_recall)
            if (keep_prec + keep_recall) > 0 else 0
        )

        # DELETE
        del_rep = s_gram_rep - c_gram_rep
        del_good = del_rep - r_gram_counter

        del_prec = (
            sum(del_good[g] / del_rep[g] for g in del_good) / len(del_rep)
            if del_rep else 0
        )

        return (add_f1 + keep_f1 + del_prec) / 3, add_f1, keep_f1, del_prec
