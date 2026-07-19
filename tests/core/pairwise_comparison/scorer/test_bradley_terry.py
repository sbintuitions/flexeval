import math

import numpy as np
import pytest

from flexeval.core.pairwise_comparison import BradleyTerryScorer
from flexeval.core.pairwise_comparison.judge.base import Winner


def test_bradley_terry_scorer() -> None:
    scorer = BradleyTerryScorer(base=math.e, scale=1.0, init_rating=1.0)
    # match_results was extracted from https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    match_results = [
        ("モデルA", "モデルB", Winner.MODEL1),
        ("モデルA", "モデルB", Winner.MODEL1),
        ("モデルA", "モデルB", Winner.MODEL2),
        ("モデルA", "モデルB", Winner.MODEL2),
        ("モデルA", "モデルB", Winner.MODEL2),
        ("モデルA", "モデルD", Winner.MODEL1),
        ("モデルA", "モデルD", Winner.MODEL2),
        ("モデルA", "モデルD", Winner.MODEL2),
        ("モデルA", "モデルD", Winner.MODEL2),
        ("モデルA", "モデルD", Winner.MODEL2),
        ("モデルB", "モデルC", Winner.MODEL1),
        ("モデルB", "モデルC", Winner.MODEL1),
        ("モデルB", "モデルC", Winner.MODEL1),
        ("モデルB", "モデルC", Winner.MODEL1),
        ("モデルB", "モデルC", Winner.MODEL1),
        ("モデルB", "モデルC", Winner.MODEL2),
        ("モデルB", "モデルC", Winner.MODEL2),
        ("モデルB", "モデルC", Winner.MODEL2),
        ("モデルC", "モデルD", Winner.MODEL1),
        ("モデルC", "モデルD", Winner.MODEL2),
        ("モデルC", "モデルD", Winner.MODEL2),
        ("モデルC", "モデルD", Winner.MODEL2),
    ]

    model_scores = scorer.compute_scores(match_results)
    assert model_scores["モデルD"] > model_scores["モデルB"] > model_scores["モデルC"] > model_scores["モデルA"]


@pytest.mark.parametrize(
    "match_results",
    [
        [("A", "B", Winner.MODEL1)],
        [
            ("A", "B", Winner.MODEL1),
            ("B", "C", Winner.MODEL1),
            ("C", "A", Winner.MODEL2),
            ("A", "D", Winner.MODEL1),
            ("B", "D", Winner.MODEL1),
            ("C", "D", Winner.MODEL1),
        ],
    ],
)
def test_bradley_terry_rejects_non_strongly_connected_results(
    match_results: list[tuple[str, str, Winner]],
) -> None:
    with pytest.raises(ValueError, match="not strongly connected"):
        BradleyTerryScorer().compute_scores(match_results)


def test_bradley_terry_returns_finite_scores_for_strongly_connected_results() -> None:
    match_results = [
        ("A", "B", Winner.MODEL1),
        ("B", "C", Winner.MODEL1),
        ("C", "A", Winner.MODEL1),
    ]

    model_scores = BradleyTerryScorer().compute_scores(match_results)

    assert set(model_scores) == {"A", "B", "C"}
    assert all(np.isfinite(score) for score in model_scores.values())


def test_bradley_terry_deprecates_eps_argument() -> None:
    with pytest.warns(FutureWarning, match="deprecated and ignored"):
        BradleyTerryScorer(eps=1e-8)
