from __future__ import annotations

import json
import os
import re
import sys
import traceback
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

import _jsonnet
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from loguru import logger

from flexeval import EvalSetup, LanguageModel
from flexeval.utils.module_utils import ConfigNameResolver

from .common import (
    CONFIG_FILE_NAME,
    METRIC_FILE_NAME,
    OUTPUTS_FILE_NAME,
    Timer,
    get_env_metadata,
    override_jsonargparse_params,
    raise_error_if_results_already_exist,
    save_json,
    save_jsonl,
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
        type=Dict[str, EvalSetup],
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
    for eval_key in params_for_eval_setups:
        for override_key, override_value in overrides_for_eval_setups[eval_key].items():
            override_jsonargparse_params(params_for_eval_setups[eval_key], override_key, override_value)
    for eval_key, eval_config in params_for_eval_setups.items():
        sys.argv += [f"--eval_setups.{eval_key}", json.dumps(eval_config)]

    # Add the current directory to sys.path
    # to enable importing modules from the directory where this script is executed.
    sys.path.append(os.environ.get("ADDITIONAL_MODULES_PATH", Path.cwd()))

    args = parser.parse_args()
    if args.eval_setup and args.eval_setups:
        msg = "You can not specify both --eval_setup and --eval_setups."
        raise ValueError(msg)

    logger.info(args)
    logger.info(f"flexeval version: {version('flexeval')}")

    config_dict = as_dict(args)  # this will be used to save the config
    args = parser.instantiate_classes(args)

    # normalize args.eval_setup or args.eval_setups into a list of tuples
    eval_setups_and_metadata: list[EvalSetup, dict[str, Any], Path] = []
    if args.eval_setup:
        save_dir = Path(args.save_dir) if args.save_dir else None
        eval_setups_and_metadata.append((args.eval_setup, config_dict["eval_setup"], save_dir))
    if args.eval_setups:
        for save_folder_name, eval_setup in args.eval_setups.items():
            save_dir = Path(args.save_dir) / save_folder_name if args.save_dir else None
            eval_setups_and_metadata.append((eval_setup, config_dict["eval_setups"][save_folder_name], save_dir))

    # run evaluation
    for eval_setup, eval_setup_config, save_dir in eval_setups_and_metadata:
        logger.info(f"Evaluating with the setup: {eval_setup_config}")

        if save_dir:
            task_config = {
                "eval_setup": eval_setup_config,
                "language_model": config_dict["language_model"],
                "save_dir": str(save_dir),
                "metadata": {
                    **get_env_metadata(),
                    **config_dict["metadata"],
                },
            }
            try:
                raise_error_if_results_already_exist(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                save_json(task_config, save_dir / CONFIG_FILE_NAME)
                logger.info(f"Saved the config to {save_dir / CONFIG_FILE_NAME}")
            except FileExistsError as e:
                if not args.force:
                    logger.info(e)
                    logger.info(f"Skip evaluation:\n{e}")
                    continue
                logger.info(
                    f"Overwriting the existing file: {save_dir / CONFIG_FILE_NAME}",
                )

                save_json(task_config, save_dir / CONFIG_FILE_NAME)

        try:
            with Timer() as timer:
                metrics, outputs = eval_setup.evaluate_lm(
                    language_model=args.language_model,
                )
            metrics["elapsed_time"] = timer.time
            logger.info(f"Elapsed time: {timer.time:.2f} sec")

            if save_dir is not None:
                save_json(metrics, save_dir / METRIC_FILE_NAME)
                logger.info(f"Saved the metrics to {save_dir / METRIC_FILE_NAME}")
                if outputs is not None:
                    save_jsonl(outputs, save_dir / OUTPUTS_FILE_NAME)
                    logger.info(f"Saved the outputs to {save_dir / OUTPUTS_FILE_NAME}")

        except Exception as e:  # noqa: BLE001
            stack_trace_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(
                f"Error in evaluation:\n{e}\n{stack_trace_str}",
            )


if __name__ == "__main__":
    main()
