from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

from flexeval.core.pairwise_comparison.judge.base import Winner

from .base import PairwiseScorer


class BradleyTerryScorer(PairwiseScorer):
    name: str = "bradley_terry"

    def __init__(
        self,
        max_iters: int = 1000,
        error_tol: float = 1e-3,
        eps: float = 1e-8,
        base: float = 10.0,
        scale: float = 400.0,
        init_rating: float = 1000.0,
    ) -> None:
        self.max_iters = max_iters
        self.error_tol = error_tol
        self.eps = eps
        self.base = base
        self.scale = scale
        self.init_rating = init_rating

    def _gen_winloss_matrix(
        self,
        match_results: list[tuple[str, str, Winner]],
    ) -> dict[str, dict[str, float]]:
        """戦績を受け取り、 `matrix[モデル1][モデル2] = <モデル1がモデル2に勝った回数>` となるような辞書を返す"""
        matrix = defaultdict(lambda: defaultdict(float))

        for model1, model2, winner in match_results:
            if winner == Winner.MODEL1:
                matrix[model1][model2] += 1.0
            elif winner == Winner.MODEL2:
                matrix[model2][model1] += 1.0
            elif winner == Winner.DRAW:
                matrix[model1][model2] += 0.5
                matrix[model2][model1] += 0.5

        return matrix

    def compute_scores(
        self,
        match_results: list[tuple[str, str, Winner]],
    ) -> dict[str, float]:
        """戦績を受け取り、Bradley-Terry model (MLE) で推定した各モデルのスコアを返す。"""
        model_names = sorted(
            {m[0] for m in match_results} | {m[1] for m in match_results},
        )
        winloss_matrix = self._gen_winloss_matrix(match_results)

        # https://jmlr.org/papers/volume24/22-1086/22-1086.pdf#page=5.50 (12)
        scores = pd.Series(np.ones(len(model_names)), index=model_names)
        for iters in range(self.max_iters):
            old_scores = scores.copy()
            for target_model in scores.keys():  # noqa: SIM118
                numer = sum(
                    [
                        (winloss_matrix[target_model][other_model] * scores[other_model])
                        / (scores[target_model] + scores[other_model])
                        for other_model in winloss_matrix[target_model]
                    ],
                )
                denom = sum(
                    [
                        (winloss_matrix[other_model][target_model]) / (scores[target_model] + scores[other_model])
                        for other_model in winloss_matrix[target_model]
                    ],
                )

                scores[target_model] = numer / (denom + self.eps)

            scores /= np.exp(np.log(scores).sum()) ** (1 / len(scores))

            if (scores - old_scores).abs().sum() < self.error_tol:
                logger.info(f" * Converged after {iters} iterations.")
                break
        else:
            logger.info(
                f" * Max iterations reached ({self.max_iters} iters).",
            )

        return (
            scores.apply(
                lambda x: self.scale / np.log(self.base) * np.log(x) + self.init_rating,
            )
            .sort_values(ascending=False)
            .to_dict()
        )
