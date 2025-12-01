from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import _jsonnet
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from loguru import logger
from pydantic.types import PositiveInt

from flexeval import EvalSetup, LanguageModel, LocalRecorder, ResultRecorder
from flexeval.utils.module_utils import ConfigNameResolver

from .common import (
    Timer,
    get_env_metadata,
    override_jsonargparse_params,
)


def as_dict(self: Namespace) -> dict[str, Any]:
    """Converts the nested namespaces into nested dictionaries.

    This is a quick fix of the original as_dict method in jsonargparse.
    """
    from jsonargparse._namespace import del_clash_mark

    dic = {}
    for key, val in vars(self).items():
        if isinstance(val, Namespace):
            val = val.as_dict()  # noqa: PLW2901
        elif isinstance(val, dict) and val != {}:
            # this elif block is modified
            val = {k: v.as_dict() if isinstance(v, Namespace) else v for k, v in val.items()}  # noqa: PLW2901
        elif isinstance(val, list) and val != [] and all(isinstance(v, Namespace) for v in val):
            val = [v.as_dict() for v in val]  # noqa: PLW2901
        dic[del_clash_mark(key)] = val
    return dic


def maybe_replace_random_seed(
    eval_setup: EvalSetup, config_content: dict, seed_increment: int
) -> tuple[EvalSetup, dict]:
    """
    Increments the random seed if the EvalSetup object has a 'random_seed' attribute.

    A new EvalSetup instance and a deep-copied config dictionary are returned
    with the updated seed value.
    """
    if hasattr(eval_setup, "random_seed"):
        new_random_seed = eval_setup.random_seed + seed_increment
        new_eval_setup = dataclasses.replace(eval_setup, random_seed=new_random_seed)
        new_config_content = copy.deepcopy(config_content)
        new_config_content["init_args"]["random_seed"] = new_random_seed
        return new_eval_setup, new_config_content
    return eval_setup, config_content


def generate_eval_entries(
    eval_setup: EvalSetup, config_content: dict, group: str | None, num_repeats: PositiveInt
) -> list:
    """
    Generates a list of evaluation entries based on repeat count and group name.

    If the number of repeats (num_repeats) is 1 or less, no run-specific metadata
    (i.e., 'runN') is appended. If it is 2 or more, each entry is indexed.

    Args:
        eval_setup (EvalSetup): The evaluation setup object.
        config_content (dict): The configuration content (dictionary) for the setup.
        group (str | None): The group name for the evaluation, e.g., aio, mt-bench, etc...
                            This forms the base of the metadata (e.g., 'group/runN').
                            If None, the metadata will only be 'runN' for repeats.
        num_repeats (int): The number of times the evaluation should be repeated.

    Returns:
        list[tuple[EvalSetup, dict, str | None]]: A list of tuples, each containing:
            (EvalSetup object, config dictionary, metadata string | None)
    """
    entries: list[tuple[EvalSetup, dict, str | None]] = []

    if num_repeats <= 0:
        msg = f"num_repeats must be positive, but {num_repeats} is given"
        raise ValueError(msg)

    if num_repeats == 1:
        metadata = group if group is not None else None
        entries.append((eval_setup, config_content, metadata))
    else:
        for index in range(num_repeats):
            metadata = f"{group}/run{index}" if group else f"run{index}"
            new_eval_setup, new_config_content = maybe_replace_random_seed(eval_setup, config_content, index)
            entries.append((new_eval_setup, new_config_content, metadata))

    return entries


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = ArgumentParser(parser_mode="jsonnet")
    parser.add_subclass_arguments(
        LanguageModel,
        nested_key="language_model",
        required=True,
        help="Language model",
    )
    parser.add_argument(
        "--eval_setup",
        type=EvalSetup,
        help="A single evaluation setup. "
        "You can specify the parameters, the path to the config file, or the name of the preset config.",
        enable_path=True,
    )
    parser.add_argument(
        "--eval_setups",
        type=dict[str, EvalSetup],
        help="A dictionary of evaluation setups. "
        "The key is the folder name where the outputs will be saved, and the value is the EvalSetup object. ",
        enable_path=True,
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
        "--retry",
        type=bool,
        default=False,
        help="Re-run only the failed evaluation runs. Skip runs if metrics are already saved.",
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
        type=dict[str, Any],
        default={},
        help="Metadata to save in config.json",
    )
    parser.add_argument(
        "--num_repeats",
        type=PositiveInt,
        default=1,
        help="Number of times to repeat the evaluation",
    )
    parser.add_argument(
        "--cleanup_after_generation",
        type=bool,
        default=False,
        help="Whether to clean up language model resources after each evaluation run. "
        "This is useful when using LLM-based metrics in the same job, as it helps release GPU memory.",
    )

    config_name_resolver = ConfigNameResolver()
    # Resolve the preset name to the path to the config file before parsing the arguments.
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--eval_setup" or re.match(r"^--eval_setups\.[^.]+$", arg):
            maybe_preset_name = sys.argv[i + 1]
            resolved_config_path = config_name_resolver(maybe_preset_name)
            if resolved_config_path is None:
                continue
            sys.argv[i + 1] = _jsonnet.evaluate_file(resolved_config_path)

    # Overrides the arguments in `--eval_setups`
    # because jsonargparse does not support override `dict[str, EvalSetup]`
    params_for_eval_setups: dict[str, dict[str, Any]] = {}
    overrides_for_eval_setups: dict[str, dict[str, str]] = defaultdict(dict)
    indices_to_pop: list[int] = []
    for i, arg in enumerate(sys.argv[:-1]):
        if re.match(r"^--eval_setups\.[^.]+$", arg):
            setup_name = arg.split(".")[1]
            params_for_eval_setups[setup_name] = json.loads(sys.argv[i + 1])
            indices_to_pop += [i, i + 1]
        elif re.match(r"^--eval_setups\.[^.]+\..*?$", arg):
            setup_name = arg.split(".")[1]
            override_key = ".".join(arg.split(".")[2:])
            overrides_for_eval_setups[setup_name][override_key] = sys.argv[i + 1]
            indices_to_pop += [i, i + 1]
    sys.argv = [a for i, a in enumerate(sys.argv) if i not in indices_to_pop]
    for eval_key in params_for_eval_setups:  # noqa: PLC0206
        for override_key, override_value in overrides_for_eval_setups[eval_key].items():
            override_jsonargparse_params(params_for_eval_setups[eval_key], override_key, override_value)
    for eval_key, eval_config in params_for_eval_setups.items():
        sys.argv += [f"--eval_setups.{eval_key}", json.dumps(eval_config)]

    # Add the current directory to sys.path
    # to enable importing modules from the directory where this script is executed.
    sys.path.append(os.environ.get("ADDITIONAL_MODULES_PATH", str(Path.cwd())))

    args = parser.parse_args()
    if args.eval_setup and args.eval_setups:
        msg = "You can not specify both --eval_setup and --eval_setups."
        raise ValueError(msg)

    logger.info(args)
    logger.info(f"flexeval version: {version('flexeval')}")

    config_dict = as_dict(args)  # this will be used to save the config

    # We instantiate the result_recorder first
    # to allow it to initialize global logging modules (e.g., wandb) that other classes might use.
    result_recorder = parser.instantiate_classes({"result_recorder": args.pop("result_recorder")}).result_recorder
    args = parser.instantiate_classes(args)
    args.result_recorder = result_recorder

    result_recorders: list[ResultRecorder] = []
    if args.save_dir:
        result_recorders.append(LocalRecorder(args.save_dir, force=args.force))
    if args.result_recorder:
        result_recorders.append(args.result_recorder)

    # normalize args.eval_setup or args.eval_setups into a list of tuples,
    # which contain (eval_setup, eval_setup_config, group)
    eval_setups_and_metadata: list[tuple[EvalSetup, dict[str, Any], str | None]] = []

    if args.eval_setup:
        eval_setups_and_metadata.extend(
            generate_eval_entries(
                eval_setup=args.eval_setup,
                config_content=config_dict["eval_setup"],
                group=None,
                num_repeats=args.num_repeats,
            )
        )

    if args.eval_setups:
        for group, eval_setup in args.eval_setups.items():
            eval_setups_and_metadata.extend(
                generate_eval_entries(
                    eval_setup=eval_setup,
                    config_content=config_dict["eval_setups"][group],
                    group=group,
                    num_repeats=args.num_repeats,
                )
            )

    exit_code = os.EX_OK
    # run evaluation
    for eval_setup, eval_setup_config, group in eval_setups_and_metadata:
        logger.info(f"Evaluating with the setup: {eval_setup_config}")

        task_config = {
            "eval_setup": eval_setup_config,
            "language_model": config_dict["language_model"],
            "metadata": {
                **get_env_metadata(),
                **config_dict["metadata"],
            },
        }

        try:
            if args.retry and not args.force:
                to_skip = all(result_recorder.is_metrics_saved(group) for result_recorder in result_recorders)
                if to_skip:
                    logger.info(f"Skipping the evaluation for group: {group} as the metrics are already saved.")
                    continue
                for result_recorder in result_recorders:
                    if hasattr(result_recorder, "force"):
                        result_recorder.force = True

            for result_recorder in result_recorders:
                result_recorder.record_config(task_config, group)

            with Timer() as timer:
                metrics, outputs = eval_setup.evaluate_lm(
                    language_model=args.language_model,
                    cleanup_after_generation=args.cleanup_after_generation,
                )
            metrics["elapsed_time"] = timer.time
            logger.info(f"Elapsed time: {timer.time:.2f} sec")

            for result_recorder in result_recorders:
                result_recorder.record_metrics(metrics, group)
                if outputs is not None:
                    result_recorder.record_model_outputs(outputs, group)

        except Exception as e:  # noqa: BLE001
            stack_trace_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(
                f"Error in evaluation:\n{e}\n{stack_trace_str}",
            )
            exit_code = os.EX_DATAERR

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
