from __future__ import annotations

from typing import Any

from flexeval.core.pairwise_comparison import PairwiseJudge, PairwiseScorer, Winner


class DummyPairwiseJudge(PairwiseJudge):
    def judge(self, model1_item: dict[str, Any], model2_item: dict[str, Any]) -> tuple[Winner, str]:
        return Winner.DRAW, "dummy"

    def batch_judge(self, batch_model_items: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[tuple[Winner, str]]:
        return [(Winner.DRAW, "dummy") for _ in batch_model_items]


class DummyPairwiseScorer(PairwiseScorer):
    def compute_scores(self, match_results: list[tuple[str, str, Winner]]) -> dict[str, float]:
        all_model_names: set[str] = set()
        for model1, model2, _ in match_results:
            all_model_names.add(model1)
            all_model_names.add(model2)
        return dict.fromkeys(all_model_names, 1.0)
