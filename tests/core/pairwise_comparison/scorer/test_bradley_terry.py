import math

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
