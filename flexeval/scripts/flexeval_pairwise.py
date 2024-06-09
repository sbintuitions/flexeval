from __future__ import annotations

import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List

from jsonargparse import ActionConfigFile, ArgumentParser

from flexeval import Match, MatchMaker, PairwiseJudge, PairwiseScorer, evaluate_pairwise

from .common import (
    CONFIG_FILE_NAME,
    METRIC_FILE_NAME,
    OUTPUTS_FILE_NAME,
    ConfigNameResolver,
    Timer,
    get_env_metadata,
    load_jsonl,
    raise_error_if_results_already_exist,
    save_json,
    save_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(parser_mode="jsonnet")
    parser.add_argument("--lm_output_paths", type=Dict[str, str], required=True)
    parser.add_argument("--judge", type=PairwiseJudge, required=True, enable_path=True)
    parser.add_argument("--match_maker", type=MatchMaker, default={"class_path": "AllCombinations"})
    parser.add_argument(
        "--scorers",
        type=List[PairwiseScorer],
        default=[{"class_path": "WinRateScorer"}, {"class_path": "BradleyTerryScorer"}],
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the outputs",
    )
    parser.add_argument(
        "--previous_outputs_path",
        type=str,
        default=None,
        help="Path to load the previous results to reuse.",
    )
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="Overwrite the save_dir if it exists",
    )
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to the config file",
    )
    parser.add_argument(
        "--metadata",
        type=Dict[str, Any],
        default={},
        help="Metadata to save in config.json",
    )

    config_preset_directory = os.environ.get(
        "PRESET_CONFIG_JUDGE_DIR",
        Path(__file__).parent.parent / "preset_configs" / "PairwiseJudge",
    )
    config_name_resolver = ConfigNameResolver(config_preset_directory)

    # Resolve the preset name to the path to the config file before parsing the arguments.
    # This is necessary when the preset name is passed with overriding arguments like
    # `--judge preset_name --judge.param value`
    # In this case, jsonargparse does not know preset_name represents a module and
    # the overriding arguments will erase the preset name.
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--judge":
            maybe_preset_name = sys.argv[i + 1]
            resolved_config_path = config_name_resolver(maybe_preset_name)
            if resolved_config_path is not None:
                sys.argv[i + 1] = resolved_config_path

    args = parser.parse_args()
    logger.info(args)

    # check if the save_dir already exists here early to avoid time-consuming instantiation
    if args.save_dir is not None and not args.force:
        raise_error_if_results_already_exist(args.save_dir)

    args_as_dict = args.as_dict()
    args_as_dict["metadata"].update(get_env_metadata())
    logger.info(f"flexeval version: {version('flexeval')}")

    args = parser.instantiate_classes(args)

    if args.save_dir is not None:
        logger.info(f"Saving the config to {Path(args.save_dir) / CONFIG_FILE_NAME}")
        save_json(args_as_dict, Path(args.save_dir) / CONFIG_FILE_NAME)

    model_items: dict[str, list[dict[str, Any]]] = {
        name: load_jsonl(path) for name, path in args.lm_output_paths.items()
    }

    cached_matches: list[Match] = []
    if args.previous_outputs_path:
        cached_matches = [Match(**result) for result in load_jsonl(args.previous_outputs_path)]

    with Timer() as timer:
        model_scores_dict, match_info_list = evaluate_pairwise(
            model_items=model_items,
            match_maker=args.match_maker,
            judge=args.judge,
            scorers=args.scorers,
            cached_matches=cached_matches,
            batch_size=args.batch_size,
        )
    logger.info(f"Elapsed time: {timer.time}")
    model_scores_dict["elapsed_time"] = timer.time

    if args.save_dir is not None:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving the model scores to {Path(args.save_dir) / METRIC_FILE_NAME}")
        save_json(model_scores_dict, Path(args.save_dir) / METRIC_FILE_NAME)

        logger.info(f"Saving the outputs to {Path(args.save_dir) / OUTPUTS_FILE_NAME}")
        save_jsonl(match_info_list, Path(args.save_dir) / OUTPUTS_FILE_NAME)


if __name__ == "__main__":
    main()
