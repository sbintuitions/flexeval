from __future__ import annotations

import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Union

import _jsonnet
from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from flexeval import ChatDataset, GenerationDataset, LocalRecorder, Metric, ResultRecorder, evaluate_from_file
from flexeval.utils.module_utils import ConfigNameResolver

from .common import (
    Timer,
    get_env_metadata,
)


def main() -> None:  # noqa: C901
    parser = ArgumentParser(parser_mode="jsonnet")
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Jsonl file containing task inputs and model outputs.",
    )
    parser.add_argument(
        "--metrics",
        type=Union[List[Metric], Metric],
        required=True,
        help="You can specify the parameters, the path to the config file, or the name of the preset config.",
    )
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
        "--eval_dataset",
        type=Union[GenerationDataset, ChatDataset],
        default=None,
        help="If specified, override the references with the ones from the generation_dataset.",
    )
    parser.add_argument(
        "--metadata",
        type=Dict[str, Any],
        default={},
        help="Metadata to save in config.json",
    )
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to the config file",
    )

    config_name_resolver = ConfigNameResolver()
    # Resolve the preset name to the path to the config file before parsing the arguments.
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--metrics":
            maybe_preset_name = sys.argv[i + 1]
            resolved_config_path = config_name_resolver(maybe_preset_name)
            if resolved_config_path is None:
                continue
            sys.argv[i + 1] = _jsonnet.evaluate_file(resolved_config_path)
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--metrics+="):
            maybe_preset_name = arg.split("=", 1)[1]
            resolved_config_path = config_name_resolver(maybe_preset_name)
            if resolved_config_path is not None:
                sys.argv[i] = "--metrics+=" + _jsonnet.evaluate_file(resolved_config_path)

    # Add the current directory to sys.path
    # to enable importing modules from the directory where this script is executed.
    sys.path.append(os.environ.get("ADDITIONAL_MODULES_PATH", Path.cwd()))

    args = parser.parse_args()
    logger.info(args)

    result_recorders: list[ResultRecorder] = []
    if args.save_dir is not None:
        result_recorders.append(LocalRecorder(args.save_dir, force=args.force))

    # normalize args.metrics to a list
    if not isinstance(args.metrics, list):
        args.metrics = [args.metrics]

    args_as_dict = args.as_dict()
    args_as_dict["metadata"].update(get_env_metadata())
    logger.info(f"flexeval version: {version('flexeval')}")

    args = parser.instantiate_classes(args)

    for result_recorder in result_recorders:
        result_recorder.record_config(args_as_dict)

    with Timer() as timer:
        metrics_summary_dict, instance_metrics_list = evaluate_from_file(
            eval_file=args.eval_file,
            metrics=args.metrics,
            eval_dataset=args.eval_dataset,
        )
    logger.info(f"Elapsed time: {timer.time}")
    metrics_summary_dict["elapsed_time"] = timer.time

    model_outputs: list[dict[str, str]] = []
    with open(args.eval_file) as f:
        for line, instance_metrics in zip(f, instance_metrics_list):
            model_outputs.append({**json.loads(line.strip()), **instance_metrics})

    for result_recorder in result_recorders:
        result_recorder.record_metrics(metrics_summary_dict)
        result_recorder.record_model_outputs(model_outputs)


if __name__ == "__main__":
    main()
