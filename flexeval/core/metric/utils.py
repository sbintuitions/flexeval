from __future__ import annotations

from collections import defaultdict
from typing import TypeVar

from flexeval.core.string_processor.base import StringProcessor

T = TypeVar("T")


def aggregate_category_wise_scores(scores: list[float | bool], categories: list[T]) -> dict[T, float]:
    """
    Compute average scores for each category.

    Args:
        scores: List of numeric scores (float or bool).
        categories: List of category labels corresponding to each score.

    Returns:
        Dictionary mapping category to average score.
    """
    if len(scores) != len(categories):
        msg = f"Length of scores ({len(scores)}) and category_keys ({len(categories)}) must be the same."
        raise ValueError(msg)

    category_scores = defaultdict(list)
    for score, category in zip(scores, categories):
        category_scores[category].append(score)

    return {category: sum(scores) / len(scores) for category, scores in category_scores.items()}


def apply_string_processors(text: str, processors: StringProcessor | list[StringProcessor] | None) -> str:
    """
    Apply string processors to a list of texts.

    Args:
        text: List of texts to process.
        processors: Single processor, list of processors, or None.

    Returns:
        List of processed texts.
    """
    if processors is None:
        return text

    if isinstance(processors, StringProcessor):
        processors = [processors]

    processed_text = text
    for processor in processors:
        processed_text = processor(processed_text)
    return processed_text


def validate_inputs(
    lm_outputs: list[str],
    references_list: list[list[str]],
    extra_info_list: list[dict[str, str]] | None = None,
) -> None:
    """
    Validate metric inputs and normalize extra_info_list.

    Args:
        lm_outputs: List of model outputs.
        references_list: List of reference lists.
        extra_info_list: List of extra info dicts or None.

    Raises:
        ValueError: If input lengths don't match.
    """
    if len(lm_outputs) != len(references_list):
        msg = (
            f"Number of model outputs ({len(lm_outputs)}) and number of references ({len(references_list)}) "
            "should be the same."
        )
        raise ValueError(msg)

    if extra_info_list is not None and len(extra_info_list) != len(lm_outputs):
        msg = (
            f"Number of extra_info entries ({len(extra_info_list)}) should match "
            f"number of outputs ({len(lm_outputs)})."
        )
        raise ValueError(msg)
