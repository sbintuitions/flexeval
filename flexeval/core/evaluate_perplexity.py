from __future__ import annotations

import logging
import math
from collections import defaultdict

from tqdm import tqdm

from .language_model import LanguageModel
from .metric.tokenizer import Tokenizer
from .text_dataset import TextDataset
from .utils.data_util import batch_iter

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    language_model: LanguageModel,
    eval_dataset: TextDataset,
    batch_size: int,
    tokenizer: Tokenizer | None = None,
) -> dict[str, float]:
    total_log_prob = 0.0

    token_counts: dict[str, int] = defaultdict(int)
    with tqdm() as pbar:
        for batch in batch_iter(eval_dataset, batch_size):
            log_probs = language_model.batch_compute_log_probs(batch)
            total_log_prob += sum(log_probs)

            for text in batch:
                token_counts["byte"] += len(text.encode("utf-8"))
                token_counts["character"] += len(text)
                if tokenizer:
                    token_counts["token"] += len(tokenizer.tokenize(text))

            pbar.update(len(batch))

    metrics_dict: dict[str, float] = {
        f"perplexity_per_{token_type}": math.exp(-total_log_prob / counts)
        for token_type, counts in token_counts.items()
    }
    metrics_dict["total_log_prob"] = total_log_prob
    logger.info(metrics_dict)
    return metrics_dict
