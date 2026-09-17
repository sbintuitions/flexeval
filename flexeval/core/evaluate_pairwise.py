from __future__ import annotations

from dataclasses import asdict
from typing import Any

from loguru import logger
from tqdm import tqdm

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


def _get_item_identity(item: dict[str, Any]) -> dict[str, Any] | None:
    """Extract model-independent fields used to align outputs from different models."""
    identity: dict[str, Any] = {}
    if "references" in item:
        identity["references"] = item["references"]

    extra_info = item.get("extra_info", item.get("task_inputs"))
    if isinstance(extra_info, dict):
        messages = extra_info.get("messages")
        if isinstance(messages, list):
            identity["messages"] = [message for message in messages if message.get("role") != "assistant"]
            other_extra_info = {key: value for key, value in extra_info.items() if key != "messages"}
            if other_extra_info:
                identity["extra_info"] = other_extra_info
        else:
            identity["extra_info"] = extra_info

    return identity or None


def _validate_model_item_alignment(model_items: dict[str, list[dict[str, Any]]]) -> None:
    model_names = sorted(model_items)
    if len(model_names) < 2:
        msg = "Pairwise evaluation requires outputs from at least two models."
        raise ValueError(msg)

    reference_model = model_names[0]
    for index in range(len(model_items[reference_model])):
        reference_identity = _get_item_identity(model_items[reference_model][index])
        for model_name in model_names[1:]:
            model_identity = _get_item_identity(model_items[model_name][index])
            if model_identity != reference_identity:
                msg = (
                    f"Model outputs are not aligned at index {index}: "
                    f"{reference_model!r} and {model_name!r} have different inputs or references."
                )
                raise ValueError(msg)


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
    _validate_model_item_alignment(model_items)

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
    if cache_count > 0:
        logger.info(f"Loaded {cache_count} judged matches from cache.")

    newly_judged_results: list[tuple[Winner, str]] = []
    with tqdm(total=len(unjudged_matches)) as pbar:
        for batch_matches in batch_iter(unjudged_matches, batch_size):
            batch_inputs = [(match.model1_item, match.model2_item) for (_, match) in batch_matches]
            newly_judged_results.extend(judge.batch_judge(batch_inputs))
            pbar.update(len(batch_matches))
    logger.info(f"Judged {len(newly_judged_results)} matches.")

    for (index, match), (winner, rationale) in zip(
        unjudged_matches,
        newly_judged_results,
    ):
        match.winner = winner
        match.rationale = rationale
        judged_matches.append((index, match))

    match_info_list: list[dict[str, Any]] = [asdict(match) for _, match in sorted(judged_matches)]

    match_results = [(i["model1"], i["model2"], i["winner"]) for i in match_info_list]
    model_scores_dict: dict[str, dict[str, float]] = {}
    for scorer in scorers:
        scorer_name = scorer.get_name()
        try:
            model_scores_dict[scorer_name] = scorer.compute_scores(match_results)
        except Exception:  # noqa: BLE001 - one scorer must not discard completed judge results
            logger.opt(exception=True).warning(
                f"Failed to compute pairwise scores with {scorer_name!r}; "
                "the scorer result will be omitted, but judged matches will still be returned."
            )
    logger.info(model_scores_dict)
    return model_scores_dict, match_info_list
