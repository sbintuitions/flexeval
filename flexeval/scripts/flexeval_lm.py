from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Union

import _jsonnet
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from flexeval import (
    ChatDataset,
    FewShotGenerator,
    GenerationDataset,
    LanguageModel,
    Metric,
    MultipleChoiceDataset,
    PromptTemplate,
    TextDataset,
    Tokenizer,
    evaluate_chat_response,
    evaluate_generation,
    evaluate_multiple_choice,
    evaluate_perplexity,
)

from .common import (
    CONFIG_FILE_NAME,
    METRIC_FILE_NAME,
    OUTPUTS_FILE_NAME,
    ConfigNameResolver,
    Timer,
    get_args_from_path,
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


class EvalSetup(ABC):
    """Abstract class to give evaluation functions a common interface."""

    @abstractmethod
    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        pass


@dataclass
class ChatResponse(EvalSetup):
    eval_dataset: ChatDataset
    gen_kwargs: dict[str, Any]
    metrics: list[Metric] | Metric | None = None
    batch_size: int = 4

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        metrics = self.metrics or []
        if isinstance(metrics, Metric):
            metrics = [metrics]

        return evaluate_chat_response(
            language_model=language_model,
            gen_kwargs=self.gen_kwargs,
            eval_dataset=self.eval_dataset,
            metrics=metrics,
            batch_size=self.batch_size,
        )


@dataclass
class Generation(EvalSetup):
    eval_dataset: GenerationDataset
    prompt_template: PromptTemplate
    gen_kwargs: dict[str, Any]
    few_shot_generator: FewShotGenerator | None = None
    metrics: list[Metric] | Metric | None = None
    batch_size: int = 4

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        metrics = self.metrics or []
        if isinstance(metrics, Metric):
            metrics = [metrics]

        return evaluate_generation(
            language_model=language_model,
            gen_kwargs=self.gen_kwargs,
            eval_dataset=self.eval_dataset,
            prompt_template=self.prompt_template,
            few_shot_generator=self.few_shot_generator,
            metrics=metrics,
            batch_size=self.batch_size,
        )


@dataclass
class MultipleChoice(EvalSetup):
    eval_dataset: MultipleChoiceDataset
    prompt_template: PromptTemplate
    few_shot_generator: FewShotGenerator | None = None
    batch_size: int = 4

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        return evaluate_multiple_choice(
            language_model=language_model,
            eval_dataset=self.eval_dataset,
            prompt_template=self.prompt_template,
            few_shot_generator=self.few_shot_generator,
            batch_size=self.batch_size,
        )


@dataclass
class Perplexity(EvalSetup):
    eval_dataset: TextDataset
    batch_size: int = 4
    tokenizer: Tokenizer | None = None

    def evaluate_lm(
        self,
        language_model: LanguageModel,
    ) -> tuple[dict[str, float], list[dict[str, Any]] | None]:
        metrics = evaluate_perplexity(
            language_model=language_model,
            eval_dataset=self.eval_dataset,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
        )
        return metrics, None


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
        type=Union[Union[EvalSetup, str], Dict[str, Union[EvalSetup, str]]],
        help="Evaluation setups. "
        "You can specify the parameters, the path to the config file, or the name of the preset config.",
        enable_path=True,
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
        "PRESET_CONFIG_EVAL_DIR",
        Path(__file__).parent.parent / "preset_configs" / "EvalSetup",
    )
    config_name_resolver = ConfigNameResolver(config_preset_directory)

    # Resolve the preset name to the path to the config file before parsing the arguments.
    # This is necessary when the preset name is passed with overriding arguments like
    # `--eval_setup preset_name --eval_setup.param value`
    # In this case, jsonargparse does not know preset_name represents a module and
    # the overriding arguments will erase the preset name.
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--eval_setup":
            maybe_preset_name = sys.argv[i + 1]
            resolved_config_path = config_name_resolver(maybe_preset_name)
            if resolved_config_path is not None:
                sys.argv[i + 1] = resolved_config_path

    args = parser.parse_args()
    logger.info(args)
    logger.info(f"flexeval version: {version('flexeval')}")

    config_dict = as_dict(args)  # this will be used to save the config

    args = parser.instantiate_classes(args)

    # normalize the format of eval_setups (a single object or a dict of objects) to a list of tuples
    eval_setups_and_metadata: list[list[EvalSetup | str, dict, Path | None]] = []
    if isinstance(args.eval_setup, dict):
        # parse the nested arguments
        overrides: dict[str, dict[str, Any]] = defaultdict(dict)
        for override_path, value in args.eval_setup.items():
            if "." in override_path:
                main_key, sub_key = override_path.split(".", 1)
                overrides[main_key][sub_key] = value

        # Parse the main arguments.
        for setup_name, eval_setup in args.eval_setup.items():
            if "." in setup_name:
                continue

            eval_config_dict = config_dict["eval_setup"][setup_name]

            # If `eval_setup` is a string, it is a preset name or a config path,
            # We need to resolve it.
            if isinstance(eval_setup, str):
                # replace eval_setup with an actual `EvalSetup` object
                eval_config_path = config_name_resolver(eval_setup)
                if eval_config_path is None:
                    msg = f"Invalid eval_setup: {eval_setup}"
                    raise ValueError(msg)
                eval_setup = instantiate_module_from_path(eval_config_path, EvalSetup, overrides[setup_name])  # noqa: PLW2901

                # replace config_dict to save with the content of the resolved config file
                eval_config_dict = as_dict(get_args_from_path(eval_config_path, EvalSetup, overrides[setup_name]))

            setup_save_dir = Path(args.save_dir) / setup_name if args.save_dir else None
            eval_setups_and_metadata.append([eval_setup, eval_config_dict, setup_save_dir])
    else:
        # When passed a single object, the preset name must have been resolved in sys.argv (see above).
        eval_setups_and_metadata.append(
            [args.eval_setup, config_dict["eval_setup"], Path(args.save_dir) if args.save_dir else None],
        )

    # If a eval_setup is specified as a preset config name, resolve the config path and instantiate the object
    for i, (eval_setup, _, _) in enumerate(eval_setups_and_metadata):
        if isinstance(eval_setup, str):
            # replace eval_setup with an actual `EvalSetup` object
            eval_config_path = config_name_resolver(eval_setup)
            if eval_config_path is None:
                msg = f"Invalid eval_setup: {eval_setup}"
                raise ValueError(msg)
            eval_setups_and_metadata[i][0] = instantiate_module_from_path(eval_config_path, EvalSetup)

            # replace config_dict to save with the content of the resolved config file
            eval_config_dict = json.loads(_jsonnet.evaluate_file(eval_config_path))
            eval_setups_and_metadata[i][1] = eval_config_dict

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

                logger.info(f"Saving the config to {save_dir / CONFIG_FILE_NAME}")
                save_dir.mkdir(parents=True, exist_ok=True)

                save_json(task_config, save_dir / CONFIG_FILE_NAME)
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
                if outputs is not None:
                    save_jsonl(outputs, save_dir / OUTPUTS_FILE_NAME)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error in evaluation:\n{e}")


if __name__ == "__main__":
    main()
