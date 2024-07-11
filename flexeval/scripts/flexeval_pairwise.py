from __future__ import annotations

import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List

import _jsonnet
from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from flexeval import LocalRecorder, Match, MatchMaker, PairwiseJudge, PairwiseScorer, ResultRecorder, evaluate_pairwise
from flexeval.utils.module_utils import ConfigNameResolver

from .common import (
    Timer,
    get_env_metadata,
    load_jsonl,
)


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
        "--previous_outputs_path",
        type=str,
        default=None,
        help="Path to load the previous results to reuse.",
    )
    # Saving arguments
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the outputs",
    )
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        help="Overwrite the save_dir if it exists",
    )
    parser.add_argument(
        "--result_recorder",
        type=ResultRecorder,
        default=None,
        help="Result recorder to save the evaluation results",
    )
    # Argument parsing arguments
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to the config file",
    )
    # Metadata
    parser.add_argument(
        "--metadata",
        type=Dict[str, Any],
        default={},
        help="Metadata to save in config.json",
    )

    config_name_resolver = ConfigNameResolver()
    # Resolve the preset name to the path to the config file before parsing the arguments.
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--judge":
            maybe_preset_name = sys.argv[i + 1]
            resolved_config_path = config_name_resolver(maybe_preset_name)
            if resolved_config_path is not None:
                sys.argv[i + 1] = _jsonnet.evaluate_file(resolved_config_path)

    # Add the current directory to sys.path
    # to enable importing modules from the directory where this script is executed.
    sys.path.append(os.environ.get("ADDITIONAL_MODULES_PATH", str(Path.cwd())))

    args = parser.parse_args()
    logger.info(args)

    args_as_dict = args.as_dict()
    args_as_dict["metadata"].update(get_env_metadata())
    logger.info(f"flexeval version: {version('flexeval')}")

    # We instantiate the result_recorder first
    # to allow it to initialize global logging modules (e.g., wandb) that other classes might use.
    result_recorder = parser.instantiate_classes({"result_recorder": args.pop("result_recorder")}).result_recorder
    args = parser.instantiate_classes(args)
    args.result_recorder = result_recorder

    result_recorders: list[ResultRecorder] = []
    if args.save_dir is not None:
        result_recorders.append(LocalRecorder(args.save_dir, force=args.force))
    if args.result_recorder:
        result_recorders.append(args.result_recorder)

    for result_recorder in result_recorders:
        result_recorder.record_config(args_as_dict)

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

    for result_recorder in result_recorders:
        result_recorder.record_metrics(model_scores_dict)
        result_recorder.record_model_outputs(match_info_list)


if __name__ == "__main__":
    main()
