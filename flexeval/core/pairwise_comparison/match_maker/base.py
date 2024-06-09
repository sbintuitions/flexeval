from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TypeVar

from flexeval.core.pairwise_comparison.match import Match

T = TypeVar("T")


class MatchMaker(ABC):
    """Generate matches between items from different models.

    The output is instances of the `Match` class.
    """

    @abstractmethod
    def generate_matches(
        self,
        model_items: dict[str, list[T]],
        cached_matches: list[Match] | None = None,
    ) -> Iterable[Match]:
        pass
