from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Union

import _jsonnet
from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from flexeval import ChatDataset, GenerationDataset, LocalRecorder, Metric, ResultRecorder, evaluate_from_data
from flexeval.utils.module_utils import ConfigNameResolver

from .common import (
    Timer,
    get_env_metadata,
)


class EvalDataLoader(ABC):
    """
    A class to load evaluation data.
    The evaluation data should be a list of dictionaries with the following keys:
    - task_inputs (dict[str, Any]): A dictionary containing the input data for the task.
    - lm_output (str): The output of the language model.
    - references (list[str]): A list of reference outputs.
    """

    @abstractmethod
    def load(self) -> list[dict[str, Any]]:
        pass


class JsonlEvalDataLoader(EvalDataLoader):
    """
    A class to load evaluation data from a jsonl file.
    """

    def __init__(self, eval_file: str) -> None:
        self.eval_file = eval_file

    def load(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with open(self.eval_file) as f:
            for line in f:
                item = json.loads(line)
                items.append(item)
        return items


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = ArgumentParser(parser_mode="jsonnet")
    parser.add_argument(
        "--eval_file",
        type=str,
        help="Jsonl file containing task inputs and model outputs.",
    )
    parser.add_argument(
        "--eval_data_loader",
        type=EvalDataLoader,
        help="A class to load evaluation data.",
    )
    parser.add_argument(
        "--metrics",
        type=Union[List[Metric], Metric],
        required=True,
        help="You can specify the parameters, the path to the config file, or the name of the preset config.",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Union[GenerationDataset, ChatDataset],
        default=None,
        help="If specified, override the references with the ones from the generation_dataset.",
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
    sys.path.append(os.environ.get("ADDITIONAL_MODULES_PATH", str(Path.cwd())))

    args = parser.parse_args()
    logger.info(args)

    if args.eval_file and args.eval_data_loader:
        msg = "You cannot specify both eval_file and eval_data_loader."
        raise ValueError(msg)
    if not args.eval_file and not args.eval_data_loader:
        msg = "You must specify either eval_file or eval_data_loader."
        raise ValueError(msg)

    # normalize args.metrics to a list
    if not isinstance(args.metrics, list):
        args.metrics = [args.metrics]

    args_as_dict = args.as_dict()
    args_as_dict["metadata"].update(get_env_metadata())
    logger.info(f"flexeval version: {version('flexeval')}")

    # We instantiate the result_recorder first
    # to allow it to initialize global logging modules (e.g., wandb) that other classes might use.
    result_recorder = parser.instantiate_classes({"result_recorder": args.pop("result_recorder")}).result_recorder
    args = parser.instantiate_classes(args)
    args.result_recorder = result_recorder

    if args.eval_file:
        eval_data = JsonlEvalDataLoader(args.eval_file).load()
    elif args.eval_data_loader:
        eval_data = args.eval_data_loader.load()
    else:
        msg = "You must specify either eval_file or eval_data_loader."
        raise ValueError(msg)

    result_recorders: list[ResultRecorder] = []
    if args.save_dir is not None:
        result_recorders.append(LocalRecorder(args.save_dir, force=args.force))
    if args.result_recorder:
        result_recorders.append(args.result_recorder)

    for result_recorder in result_recorders:
        result_recorder.record_config(args_as_dict)

    with Timer() as timer:
        metrics_summary_dict, instance_metrics_list = evaluate_from_data(
            eval_data=eval_data,
            metrics=args.metrics,
            eval_dataset=args.eval_dataset,
        )
    logger.info(f"Elapsed time: {timer.time}")
    metrics_summary_dict["elapsed_time"] = timer.time

    model_outputs: list[dict[str, str]] = []
    for item, instance_metrics in zip(eval_data, instance_metrics_list):
        model_outputs.append({**item, **instance_metrics})

    for result_recorder in result_recorders:
        result_recorder.record_metrics(metrics_summary_dict)
        result_recorder.record_model_outputs(model_outputs)


if __name__ == "__main__":
    main()
