from __future__ import annotations

import json
import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Union

import _jsonnet
from jsonargparse import ActionConfigFile, ArgumentParser

from flexeval import ChatDataset, GenerationDataset, Metric, evaluate_from_file

from .common import (
    CONFIG_FILE_NAME,
    METRIC_FILE_NAME,
    OUTPUTS_FILE_NAME,
    ConfigNameResolver,
    Timer,
    get_env_metadata,
    instantiate_module_from_path,
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
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Jsonl file containing task inputs and model outputs.",
    )
    parser.add_argument(
        "--metrics",
        type=Union[List[Union[Metric, str]], Union[Metric, str]],
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

    args = parser.parse_args()
    logger.info(args)

    # check if the save_dir already exists here early to avoid time-consuming instantiation
    if args.save_dir is not None and not args.force:
        raise_error_if_results_already_exist(args.save_dir)

    # normalize args.metrics to a list
    if not isinstance(args.metrics, list):
        args.metrics = [args.metrics]

    args_as_dict = args.as_dict()
    args_as_dict["metadata"].update(get_env_metadata())
    logger.info(f"flexeval version: {version('flexeval')}")

    args = parser.instantiate_classes(args)

    config_preset_directory = os.environ.get(
        "PRESET_CONFIG_METRIC_DIR",
        Path(__file__).parent.parent / "preset_configs" / "Metric",
    )
    config_name_resolver = ConfigNameResolver(config_preset_directory)
    for i, metric in enumerate(args.metrics):
        if isinstance(metric, str):
            metric_config_path = config_name_resolver(metric)
            if metric_config_path is None:
                msg = f"Invalid metric: {metric}"
                raise ValueError(msg)
            instantiated_metric = instantiate_module_from_path(metric_config_path, Metric)
            args.metrics[i] = instantiated_metric
            args_as_dict["metrics"][i] = json.loads(_jsonnet.evaluate_file(metric_config_path))

    if args.save_dir is not None:
        logger.info(f"Saving the config to {Path(args.save_dir) / CONFIG_FILE_NAME}")
        save_json(args_as_dict, Path(args.save_dir) / CONFIG_FILE_NAME)

    with Timer() as timer:
        metrics_summary_dict, instance_metrics_list = evaluate_from_file(
            eval_file=args.eval_file,
            metrics=args.metrics,
            eval_dataset=args.eval_dataset,
        )
    logger.info(f"Elapsed time: {timer.time}")
    metrics_summary_dict["elapsed_time"] = timer.time

    if args.save_dir is not None:
        logger.info(f"Saving the metrics to {Path(args.save_dir) / METRIC_FILE_NAME}")
        save_json(metrics_summary_dict, Path(args.save_dir) / METRIC_FILE_NAME)

        logger.info(f"Saving the outputs to {Path(args.save_dir) / OUTPUTS_FILE_NAME}")
        with open(args.eval_file) as f:
            save_jsonl(
                (
                    {**json.loads(line.strip()), **instance_metrics}
                    for line, instance_metrics in zip(f, instance_metrics_list)
                ),
                Path(args.save_dir) / OUTPUTS_FILE_NAME,
            )


if __name__ == "__main__":
    main()
