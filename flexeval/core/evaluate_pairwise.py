from __future__ import annotations

from dataclasses import asdict
from typing import Any

from loguru import logger

from .pairwise_comparison import (
    AllCombinations,
    BradleyTerryScorer,
    Match,
    MatchMaker,
    PairwiseJudge,
    PairwiseScorer,
    Winner,
    WinRateScorer,
)
from .utils.data_util import batch_iter


def evaluate_pairwise(
    model_items: dict[str, list[dict[str, Any]]],
    judge: PairwiseJudge,
    match_maker: MatchMaker | None = None,
    scorers: list[PairwiseScorer] | None = None,
    cached_matches: list[Match] | None = None,
    batch_size: int = 4,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    match_maker = match_maker or AllCombinations()
    scorers = scorers or [WinRateScorer(), BradleyTerryScorer()]

    if len({len(items) for items in model_items.values()}) != 1:
        msg = "The number of items in the model outputs should be the same."
        raise ValueError(msg)

    # Separate matches: in cache vs. not in cache and perform judge on the latter.
    judged_matches: list[tuple[int, Match]] = []  # Store indices to restore original order.
    unjudged_matches: list[tuple[int, Match]] = []
    for index, match in enumerate(
        match_maker.generate_matches(model_items, cached_matches),
    ):
        if match.is_judged():
            judged_matches.append((index, match))
        else:
            unjudged_matches.append((index, match))
    cache_count: int = len(judged_matches)

    newly_judged_results: list[tuple[Winner, str]] = []
    for batch_matches in batch_iter(unjudged_matches, batch_size):
        batch_inputs = [(match.model1_item, match.model2_item) for (_, match) in batch_matches]
        newly_judged_results.extend(judge.batch_judge(batch_inputs))
    newly_judged_count: int = len(newly_judged_results)

    for (index, match), (winner, rationale) in zip(
        unjudged_matches,
        newly_judged_results,
    ):
        match.winner = winner
        match.rationale = rationale
        judged_matches.append((index, match))

    match_info_list: list[dict[str, Any]] = [asdict(match) for _, match in sorted(judged_matches)]

    model_scores_dict: dict[str, dict[str, float]] = {
        scorer.get_name(): scorer.compute_scores(
            [(i["model1"], i["model2"], i["winner"]) for i in match_info_list],
        )
        for scorer in scorers
    }
    logger.info(
        f"newly judged: {newly_judged_count} items, loaded from cache: {cache_count} items",
    )
    logger.info(model_scores_dict)
    return model_scores_dict, match_info_list
