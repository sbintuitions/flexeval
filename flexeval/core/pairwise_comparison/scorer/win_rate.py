from __future__ import annotations

from collections import defaultdict

from flexeval.core.pairwise_comparison.judge.base import Winner

from .base import PairwiseScorer


class WinRateScorer(PairwiseScorer):
    name: str = "win_rate"

    def compute_scores(
        self,
        match_results: list[tuple[str, str, Winner]],
    ) -> dict[str, float]:
        """戦績を受け取り、各モデルの勝率を返す。"""
        match_count_dict: dict[str, float] = defaultdict(float)
        win_count_dict: dict[str, float] = defaultdict(float)

        for model1, model2, winner in match_results:
            match_count_dict[model1] += 1
            match_count_dict[model2] += 1
            if winner == Winner.MODEL1:
                win_count_dict[model1] += 1
            elif winner == Winner.MODEL2:
                win_count_dict[model2] += 1
            elif winner == Winner.DRAW:
                win_count_dict[model1] += 0.5
                win_count_dict[model2] += 0.5

        win_rate_dict = {}
        for model in match_count_dict:
            win_rate_dict[model] = 100 * win_count_dict.get(model, 0.0) / match_count_dict[model]

        return dict(sorted(win_rate_dict.items(), key=lambda x: -x[1]))
