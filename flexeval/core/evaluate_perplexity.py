from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from loguru import logger
from tqdm import tqdm

from .language_model import LanguageModel
from .text_dataset import TextDataset, TextInstance
from .tokenizer import Tokenizer
from .utils.data_util import batch_iter


def evaluate_perplexity(
    language_model: LanguageModel,
    eval_dataset: TextDataset,
    batch_size: int,
    max_instances: int | None = None,
    tokenizer: Tokenizer | None = None,
) -> dict[str, float]:
    total_log_prob = 0.0

    eval_instances: Sequence[TextInstance] = eval_dataset
    if max_instances is not None:
        eval_instances = [eval_dataset[i] for i in range(min(max_instances, len(eval_dataset)))]

    token_counts: dict[str, int] = defaultdict(int)
    with tqdm(total=len(eval_instances)) as pbar:
        for batch in batch_iter(eval_instances, batch_size):
            log_probs = language_model.batch_compute_log_probs(
                text_list=[i.text for i in batch], prefix_list=[i.prefix for i in batch]
            )
            total_log_prob += sum(log_probs)

            for instance in batch:
                token_counts["byte"] += len(instance.text.encode("utf-8"))
                token_counts["character"] += len(instance.text)
                if tokenizer:
                    token_counts["token"] += len(tokenizer.tokenize(instance.text))

            pbar.update(len(batch))

    metrics_dict: dict[str, float] = {
        f"perplexity_per_{token_type}": math.exp(-total_log_prob / counts)
        for token_type, counts in token_counts.items()
    }
    metrics_dict["total_log_prob"] = total_log_prob
    logger.info(metrics_dict)
    return metrics_dict
