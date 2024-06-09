from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TypeVar

from flexeval.core.pairwise_comparison.judge.base import Winner

T = TypeVar("T")


@dataclass
class Match:
    model1: str
    model1_item: T
    model2: str
    model2_item: T
    winner: Winner | str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.winner, str):
            self.winner = Winner(self.winner)

    def is_judged(self) -> bool:
        return isinstance(self.winner, Winner)

    def get_key_for_cache(self) -> int:
        return hash(json.dumps([self.model1, self.model1_item, self.model2, self.model2_item]))

    def __hash__(self) -> int:
        return hash(json.dumps(asdict(self)))

    def __eq__(self, other: Match) -> bool:
        return self.__hash__() == other.__hash__()
