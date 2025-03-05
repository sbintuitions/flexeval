from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def aggregate_category_wise_scores(scores: list[float, bool], categories: list[T]) -> dict[T, float]:
    """
    Compute average scores for each category.
    """
    if len(scores) != len(categories):
        msg = f"Length of scores ({len(scores)}) and category_keys ({len(categories)}) should be the same."
        raise ValueError(msg)

    category_set = sorted(set(categories))
    category_scores = {category: [] for category in category_set}
    for score, category in zip(scores, categories):
        category_scores[category].append(score)

    return {category: sum(scores) / len(scores) for category, scores in category_scores.items()}
