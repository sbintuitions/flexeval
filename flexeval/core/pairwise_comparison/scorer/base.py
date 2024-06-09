from __future__ import annotations

from abc import ABC, abstractmethod

from flexeval.core.pairwise_comparison.judge.base import Winner


class PairwiseScorer(ABC):
    """Compute scores for each model given the match results.

    Each match result is a triple of two model names and the winner.
    """

    name: str = None

    @abstractmethod
    def compute_scores(
        self: PairwiseScorer,
        match_results: list[tuple[str, str, Winner]],
    ) -> dict[str, float]:
        pass

    @classmethod
    def get_name(cls: type[PairwiseScorer]) -> str:
        return cls.name if cls.name else cls.__name__
