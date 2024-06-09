from __future__ import annotations

import itertools
import random
from typing import Iterable

from flexeval.core.pairwise_comparison.match import Match

from .base import MatchMaker, T


class RandomCombinations(MatchMaker):
    def __init__(self, n: int = 100, incremental: bool = False, seed: int = 42) -> None:
        self.n = n
        self.incremental = incremental
        self.seed = seed

    def generate_matches(
        self,
        model_items: dict[str, list[T]],
        cached_matches: list[Match] | None = None,
    ) -> Iterable[Match]:
        model_names = sorted(model_items.keys())
        all_permutations = list(itertools.permutations(model_names, 2))

        cached_matches = cached_matches or []
        cache_dict = {match.get_key_for_cache(): match for match in cached_matches}
        model_match_counter: dict[str, int] = {name: 0 for name in model_names}
        possible_new_matches: list[Match] = []
        matches: list[Match] = []
        for m1, m2 in all_permutations:
            for item1, item2 in zip(model_items[m1], model_items[m2]):
                match = Match(m1, item1, m2, item2)
                if cached_match := cache_dict.get(match.get_key_for_cache()):
                    matches.append(cached_match)
                    model_match_counter[m1] += 1
                    model_match_counter[m2] += 1
                else:
                    possible_new_matches.append(Match(m1, item1, m2, item2))

        # If `self.incremental` is `True`, add n more matches in addition to the cached data.
        max_matches = self.n + len(matches) if self.incremental else self.n

        random.seed(self.seed)

        # For each iteration, assign the model with the fewest matches to a new match.
        while (len(matches) < max_matches) and (len(possible_new_matches) > 0):
            target_model = min(model_match_counter, key=model_match_counter.get)
            candidate_matches = [
                (i, match)
                for i, match in enumerate(possible_new_matches)
                if target_model in (match.model1, match.model2)
            ]
            index, selected_match = random.choice(candidate_matches)
            matches.append(selected_match)
            del possible_new_matches[index]
            model_match_counter[selected_match.model1] += 1
            model_match_counter[selected_match.model2] += 1

        for match in matches:
            yield match
