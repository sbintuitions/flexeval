from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from .chat_dataset import ChatDataset, ChatInstance
from .language_model import LanguageModel
from .metric import Metric
from .utils.data_util import batch_iter

logger = logging.getLogger(__name__)


def evaluate_chat_response(
    language_model: LanguageModel,
    gen_kwargs: dict[str, Any],
    eval_dataset: ChatDataset,
    metrics: list[Metric],
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    logger.info(f"Evaluate the model with gen_kwargs: {gen_kwargs}")

    all_messages_list: list[list[dict[str, str]]] = []
    references_list: list[list[str]] = []
    extra_info_list: list[dict[str, Any]] = []
    with tqdm(total=len(eval_dataset)) as pbar:
        for i, batch in enumerate(batch_iter(eval_dataset, batch_size)):
            batch: list[ChatInstance]

            input_messages_list = [chat_instance.messages for chat_instance in batch]
            if not eval_dataset.require_incremental_response():
                lm_outputs = language_model.batch_generate_chat_response(
                    input_messages_list,
                    **gen_kwargs,
                )
                for input_messages, lm_output in zip(input_messages_list, lm_outputs):
                    all_messages_list.append(
                        [*input_messages, {"role": "assistant", "content": lm_output}],
                    )
            else:
                max_num_turns = max(len(messages) for messages in input_messages_list)
                current_chat_history: list[list[dict[str, str]]] = [[] for _ in input_messages_list]
                # perform generation for each turn
                for turn in range(max_num_turns):
                    batch_ids_fed_to_model = [
                        b_id for b_id, messages in enumerate(input_messages_list) if turn < len(messages)
                    ]
                    current_model_inputs = [
                        current_chat_history[b_id] + [input_messages_list[b_id][turn]]
                        for b_id in batch_ids_fed_to_model
                    ]
                    lm_outputs = language_model.batch_generate_chat_response(
                        current_model_inputs,
                        **gen_kwargs,
                    )
                    for o_id, b_id in enumerate(batch_ids_fed_to_model):
                        current_chat_history[b_id].append(input_messages_list[b_id][turn])
                        current_chat_history[b_id].append(
                            {"role": "assistant", "content": lm_outputs[o_id]},
                        )
                all_messages_list += current_chat_history

            references_list += [chat_instance.references for chat_instance in batch]
            extra_info_list += [chat_instance.extra_info for chat_instance in batch]

            if i == 0:
                logger.info("Example of the conversation")
                logger.info(f"{all_messages_list[0]}")

            pbar.update(len(batch))
    metrics_summary_dict: dict[str, float] = {}
    instance_metrics_list: list[dict[str, Any]] = [{} for _ in range(len(all_messages_list))]
    for metric in metrics:
        metric_result = metric.evaluate(
            lm_outputs=[messages[-1]["content"] for messages in all_messages_list],
            references_list=references_list,
            task_inputs_list=[
                {"messages": messages[:-1], **extra_info}
                for messages, extra_info in zip(all_messages_list, extra_info_list)
            ],
        )

        metrics_summary_dict.update(metric_result.summary)

        if metric_result.instance_details:
            for instance_idx, instance_details in enumerate(
                metric_result.instance_details,
            ):
                instance_metrics_list[instance_idx].update(instance_details)

    logger.info(metrics_summary_dict)

    outputs = [
        {
            "lm_output": messages[-1]["content"],
            "task_inputs": {"messages": messages[:-1], **extra_info},
            "references": references,
            **instance_metrics,
        }
        for messages, references, extra_info, instance_metrics in zip(
            all_messages_list,
            references_list,
            extra_info_list,
            instance_metrics_list,
        )
    ]
    return metrics_summary_dict, outputs
