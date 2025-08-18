from __future__ import annotations

from collections import Counter
from typing import Literal

from flexeval.core.language_model.base import LMOutput
from flexeval.core.metric.base import Metric, MetricResult
from flexeval.core.metric.utils import aggregate_category_wise_scores, apply_string_processors, validate_inputs
from flexeval.core.string_processor.base import StringProcessor
from flexeval.core.string_processor.lower import StringLower
from flexeval.core.tokenizer.base import Tokenizer
from flexeval.core.tokenizer.sacrebleu_tokenizer import SacreBleuTokenizer

from .utils import extract_text_from_outputs


def to_ngram(words: list[str], n: int) -> list[str]:
    return ["__".join(words[i : i + n]) for i in range(len(words) - n + 1)]


DEFAULT_STRING_PROCESSOR = StringLower()


class SARI(Metric):
    """An implementation of SARI, a metric for evaluating text simplification.

    Based on the original implementation [1], modified to allow configurable settings
    for the maximum n-gram size and tokenizer.
    Additionally, it fixes a bug present in the original implementation [2].
    When used with the default parameters, it produces scores that are
    consistent with the HuggingFace/evaluate implementation [3].

    [1] https://github.com/cocoxu/simplification/blob/master/SARI.py
    [2] https://github.com/cocoxu/simplification/issues/6
    [3] https://huggingface.co/spaces/evaluate-metric/sari/blob/main/sari.py

    Args:
        tokenizer: An instance of `Tokenizer` to tokenize the input and output strings.
        max_ngrams: The maximum n-gram order to consider. Defaults to `4`.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in extra_info.
        lm_output_processor:
            StringProcessor or a list of StringProcessor to be applied to the model outputs before comparison.
        reference_processor: StringProcessor or list of StringProcessor to apply to the references before comparison.
        source_processor: StringProcessor or list of StringProcessor to apply to the source sentences before comparison.

    Examples:
        >>> from flexeval import SARI
        >>> sari_scorer = SARI(source_key="source")
        >>> lm_outputs = ["About 95 you now get in."]
        >>> references_list = [["About 95 species are currently known.", "About 95 species are now accepted.", "95 species are now accepted."]]
        >>> extra_info_list = [{"source": "About 95 species are currently accepted."}]
        >>> result = sari_scorer.evaluate(lm_outputs, references_list, extra_info_list)
        >>> print(result)
        MetricResult(
            summary={
                'sari_score': 0.2695360195360195,
                'sari_add': 0.08333333333333333,
                'sari_keep': 0.22527472527472525,
                'sari_del': 0.5
            },
            instance_details=[{'sari_score': 0.2695360195360195, 'sari_add': 0.08333333333333333, 'sari_keep': 0.22527472527472525, 'sari_del': 0.5}]
        )
    """  # noqa: E501

    def __init__(
        self,
        source_key: str,
        tokenizer: Tokenizer | Literal["default"] = "default",
        max_ngrams: int = 4,
        category_key: str | None = None,
        source_processor: StringProcessor | list[StringProcessor] | None = DEFAULT_STRING_PROCESSOR,
        lm_output_processor: StringProcessor | list[StringProcessor] | None = DEFAULT_STRING_PROCESSOR,
        reference_processor: StringProcessor | list[StringProcessor] | None = DEFAULT_STRING_PROCESSOR,
    ) -> None:
        if tokenizer == "default":
            tokenizer = SacreBleuTokenizer("13a")
        self._tokenizer = tokenizer
        self.source_key = source_key
        self.max_ngrams = max_ngrams
        self.category_key = category_key

        self.source_processors = source_processor
        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)

        lm_outputs = extract_text_from_outputs(lm_outputs)

        if extra_info_list is None:
            msg = "SARI requires extra_info_list"
            raise ValueError(msg)
        sources = [extra_info[self.source_key] for extra_info in extra_info_list]

        # Normalize text data
        sources = [apply_string_processors(src, self.source_processors) for src in sources]
        lm_outputs = [apply_string_processors(output, self.lm_output_processors) for output in lm_outputs]
        references_list = [
            [apply_string_processors(ref, self.reference_processors) for ref in references]
            for references in references_list
        ]

        # Compute metrics
        sari_instance_list = [
            self._calc_sentence_sari(source, lm_output, references)
            for source, lm_output, references in zip(sources, lm_outputs, references_list)
        ]

        metric_name2scores = {
            name: [s[name] for s in sari_instance_list] for name in ["sari_score", "sari_add", "sari_keep", "sari_del"]
        }

        num_instances = len(sari_instance_list)
        summary = {
            metric_name: sum(score_list) / num_instances for metric_name, score_list in metric_name2scores.items()
        }

        if self.category_key:
            categories = [extra_info[self.category_key] for extra_info in extra_info_list]
            for metric_name, score_list in metric_name2scores.items():
                category_wise_scores = aggregate_category_wise_scores(score_list, categories)
                for category, category_wise_score in category_wise_scores.items():
                    summary[f"{metric_name}/{category}"] = category_wise_score

        return MetricResult(
            summary,
            instance_details=sari_instance_list,
        )

    def _calc_sentence_sari(self, source: str, lm_output: str, references: list[str]) -> dict[str, float]:
        s_words = self._tokenizer.tokenize(source)
        c_words = self._tokenizer.tokenize(lm_output)
        r_words_list = [self._tokenizer.tokenize(reference) for reference in references]

        sari_score, sari_add, sari_keep, sari_del = 0.0, 0.0, 0.0, 0.0
        for n in range(1, self.max_ngrams + 1):
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

        add_prec = len(add_good) / len(add_grams) if add_grams else 1
        add_recall = len(add_good) / len(add_all) if add_all else 1
        add_f1 = 2 * add_prec * add_recall / (add_prec + add_recall) if (add_prec + add_recall) > 0 else 0

        # KEEP
        keep_rep = s_gram_rep & c_gram_rep
        keep_good = keep_rep & r_gram_counter
        keep_all = s_gram_rep & r_gram_counter

        keep_prec = sum(keep_good[g] / keep_rep[g] for g in keep_good) / len(keep_rep) if keep_rep else 1
        keep_recall = sum(keep_good[g] for g in keep_good) / sum(keep_all.values()) if keep_all else 1
        keep_f1 = 2 * keep_prec * keep_recall / (keep_prec + keep_recall) if (keep_prec + keep_recall) > 0 else 0

        # DELETE
        del_rep = s_gram_rep - c_gram_rep
        del_good = del_rep - r_gram_counter

        del_prec = sum(del_good[g] / del_rep[g] for g in del_good) / len(del_rep) if del_rep else 1

        return (add_f1 + keep_f1 + del_prec) / 3, add_f1, keep_f1, del_prec
