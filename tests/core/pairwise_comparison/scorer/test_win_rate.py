import pytest

from flexeval.core.pairwise_comparison import WinRateScorer
from flexeval.core.pairwise_comparison.judge.base import Winner


def test_bradley_terry_scorer() -> None:
    scorer = WinRateScorer()
    match_results = [
        ("モデルA", "モデルB", Winner.MODEL1),
        ("モデルB", "モデルC", Winner.MODEL1),
        ("モデルC", "モデルA", Winner.MODEL2),
        ("モデルD", "モデルE", Winner.DRAW),
    ]

    actual = scorer.compute_scores(match_results)
    expected = {"モデルA": 100.0, "モデルB": 50.0, "モデルC": 0.0, "モデルD": 50.0, "モデルE": 50.0}

    assert actual == pytest.approx(expected)
