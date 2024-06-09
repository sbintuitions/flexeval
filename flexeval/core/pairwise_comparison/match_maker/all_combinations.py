from __future__ import annotations

import itertools
from typing import Iterable

from flexeval.core.pairwise_comparison.match import Match

from .base import MatchMaker, T


class AllCombinations(MatchMaker):
    def __init__(self, include_reversed: bool = True) -> None:
        self.include_reversed = include_reversed

    def generate_matches(self, model_items: dict[str, list[T]], cached_matches: list[Match] | None) -> Iterable[Match]:
        model_names = sorted(model_items.keys())
        all_combinations = list(itertools.combinations(model_names, 2))

        cached_matches = cached_matches or []
        cache_dict = {match.get_key_for_cache(): match for match in cached_matches}

        if self.include_reversed:
            all_combinations += [(m2, m1) for m1, m2 in all_combinations]

        for m1, m2 in all_combinations:
            for item1, item2 in zip(model_items[m1], model_items[m2]):
                match = Match(m1, item1, m2, item2)
                if cached_match := cache_dict.get(match.get_key_for_cache()):
                    yield cached_match
                else:
                    yield match
