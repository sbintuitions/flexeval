from __future__ import annotations

import warnings
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
        eps: float | None = None,
        base: float = 10.0,
        scale: float = 400.0,
        init_rating: float = 1000.0,
    ) -> None:
        self.max_iters = max_iters
        self.error_tol = error_tol
        if eps is not None:
            warnings.warn(
                "The 'eps' argument is deprecated and ignored; it will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
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

    @staticmethod
    def _validate_strong_connectivity(
        model_names: list[str],
        winloss_matrix: dict[str, dict[str, float]],
    ) -> None:
        """Validate that the directed win graph has a finite Bradley-Terry MLE."""
        if len(model_names) < 2:
            msg = "Bradley-Terry scoring requires results for at least two models."
            raise ValueError(msg)

        def reachable(start: str, reverse: bool = False) -> set[str]:
            visited: set[str] = set()
            stack = [start]
            while stack:
                model = stack.pop()
                if model in visited:
                    continue
                visited.add(model)
                for other_model in model_names:
                    weight = winloss_matrix[other_model][model] if reverse else winloss_matrix[model][other_model]
                    if weight > 0 and other_model not in visited:
                        stack.append(other_model)
            return visited

        start = model_names[0]
        if len(reachable(start)) != len(model_names) or len(reachable(start, reverse=True)) != len(model_names):
            msg = (
                "Bradley-Terry maximum-likelihood scores are not finite because the directed win graph is not "
                "strongly connected. Add comparisons that connect the graph in both directions or use a "
                "regularized scorer."
            )
            raise ValueError(msg)

    def compute_scores(
        self,
        match_results: list[tuple[str, str, Winner]],
    ) -> dict[str, float]:
        """戦績を受け取り、Bradley-Terry model (MLE) で推定した各モデルのスコアを返す。"""
        model_names = sorted(
            {m[0] for m in match_results} | {m[1] for m in match_results},
        )
        winloss_matrix = self._gen_winloss_matrix(match_results)
        self._validate_strong_connectivity(model_names, winloss_matrix)

        # https://jmlr.org/papers/volume24/22-1086/22-1086.pdf#page=5.50 (12)
        scores = pd.Series(np.ones(len(model_names)), index=model_names)
        for iters in range(self.max_iters):
            old_scores = scores.copy()
            for target_model in scores.keys():  # noqa: SIM118
                opponents = [
                    other_model
                    for other_model in model_names
                    if other_model != target_model
                    and (winloss_matrix[target_model][other_model] > 0 or winloss_matrix[other_model][target_model] > 0)
                ]
                numer = sum(
                    [
                        (winloss_matrix[target_model][other_model] * scores[other_model])
                        / (scores[target_model] + scores[other_model])
                        for other_model in opponents
                    ],
                )
                denom = sum(
                    [
                        (winloss_matrix[other_model][target_model]) / (scores[target_model] + scores[other_model])
                        for other_model in opponents
                    ],
                )

                if numer <= 0 or denom <= 0:
                    msg = "Bradley-Terry iteration reached a non-finite boundary despite a strongly connected graph."
                    raise RuntimeError(msg)
                scores[target_model] = numer / denom

            scores /= np.exp(np.log(scores).mean())

            if not np.isfinite(scores).all():
                msg = "Bradley-Terry iteration produced non-finite scores."
                raise RuntimeError(msg)

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
