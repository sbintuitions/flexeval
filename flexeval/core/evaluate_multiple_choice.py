from __future__ import annotations

from typing import Any, Sequence

from loguru import logger
from tqdm import tqdm

from .few_shot_generator import FewShotGenerator
from .language_model import LanguageModel
from .multiple_choice_dataset import MultipleChoiceDataset, MultipleChoiceInstance
from .prompt_template import PromptTemplate
from .utils.data_util import batch_iter


def evaluate_multiple_choice(
    language_model: LanguageModel,
    eval_dataset: MultipleChoiceDataset,
    prompt_template: PromptTemplate,
    batch_size: int,
    max_instances: int | None = None,
    few_shot_generator: FewShotGenerator | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    eval_instances: Sequence[MultipleChoiceInstance] = eval_dataset
    if max_instances is not None:
        eval_instances = [eval_dataset[i] for i in range(min(max_instances, len(eval_dataset)))]

    results: list[dict[str, Any]] = []
    with tqdm(total=len(eval_instances)) as pbar:
        for batch_id, batch in enumerate(batch_iter(eval_instances, batch_size)):
            batch: list[MultipleChoiceInstance]

            batch_prefixes: list[str] = []
            batch_choices: list[str] = []
            for eval_instance in batch:
                template_inputs = {**eval_instance.inputs, "choices": eval_instance.choices}

                if few_shot_generator is not None:
                    few_shot_instances = few_shot_generator(template_inputs)
                    few_shot_item_list: list[dict[str, Any]] = []
                    for few_shot_instance in few_shot_instances:
                        if isinstance(few_shot_instance, MultipleChoiceInstance):
                            few_shot_item = {
                                **few_shot_instance.inputs,
                                "choices": few_shot_instance.choices,
                                "answer_index": few_shot_instance.answer_index,
                            }
                            few_shot_item_list.append(few_shot_item)
                        else:
                            msg = f"Invalid instance type: {type(few_shot_instance)}"
                            raise TypeError(msg)
                    template_inputs = {**template_inputs, "few_shot_data": few_shot_item_list}

                prefix = prompt_template.embed_inputs(template_inputs)
                batch_prefixes += [prefix] * len(eval_instance.choices)
                batch_choices += eval_instance.choices

            if batch_id == 0:
                logger.info("Example of the model inputs and outputs:")
                logger.info(f"prefix: {batch_prefixes[0]}")
                logger.info(f"choices: {batch_choices[:len(eval_instance.choices)]}")

            batch_log_probs = language_model.batch_compute_log_probs(
                text_list=batch_choices,
                prefix_list=batch_prefixes,
            )

            # calculate accuracy
            i = 0
            for eval_instance in batch:
                log_probs_for_choices = batch_log_probs[i : i + len(eval_instance.choices)]
                # select the choice with the highest log probability as model output
                max_log_prob = max(log_probs_for_choices)
                max_log_prob_index = log_probs_for_choices.index(max_log_prob)

                # we also calculate accuracy using byte-normalized log probabilities
                # for the discussion on normalization methods, see
                # https://github.com/EleutherAI/lm-evaluation-harness/issues/1396
                # https://blog.eleuther.ai/multiple-choice-normalization/
                norm_log_probs = [
                    log_p / len(choice.encode("utf-8"))
                    for log_p, choice in zip(log_probs_for_choices, eval_instance.choices)
                ]
                max_norm_log_p = max(norm_log_probs)
                max_norm_log_p_index = norm_log_probs.index(max_norm_log_p)

                results.append(
                    {
                        "prefix": batch_prefixes[i],
                        "choices": eval_instance.choices,
                        "answer_index": eval_instance.answer_index,
                        "log_probs": log_probs_for_choices,
                        "prediction": max_log_prob_index,
                        "byte_norm_log_probs": norm_log_probs,
                        "byte_norm_prediction": max_norm_log_p_index,
                    },
                )
                i += len(eval_instance.choices)

            pbar.update(len(batch))

    accuracy = sum(res["prediction"] == res["answer_index"] for res in results) / len(results)
    byte_norm_accuracy = sum(res["byte_norm_prediction"] == res["answer_index"] for res in results) / len(results)

    metrics_dict: dict[str, float] = {
        "accuracy": accuracy,
        "byte_norm_accuracy": byte_norm_accuracy,
    }
    logger.info(metrics_dict)
    return metrics_dict, results
