from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from loguru import logger
from tqdm import tqdm

from flexeval.core.language_model.base import LMOutput

from .chat_dataset import ChatDataset, ChatInstance
from .few_shot_generator import FewShotGenerator
from .language_model import LanguageModel
from .metric import Metric
from .utils.data_util import batch_iter


def _remove_redundant_keys_from_messages(
    messages: list[dict[str, str]],
    remove_keys: Iterable[str],
) -> list[dict[str, str]]:
    """
    Remove specified keys from all turns in `messages`.

    Some keys such as `finish_reason` are added in `evaluate_chat_response()` for logging.
    However, some APIs of Azure OpenAI (e.g. tsuzumi-7b-instruct) do not allow extra input keys.
    Thus this removal is required before each input step.
    """
    remove_keys = set(remove_keys)
    return [{key: value for key, value in message.items() if key not in remove_keys} for message in messages]


def evaluate_chat_response(  # noqa: C901,PLR0912
    language_model: LanguageModel,
    gen_kwargs: dict[str, Any],
    eval_dataset: ChatDataset,
    metrics: list[Metric],
    batch_size: int,
    max_instances: int | None = None,
    few_shot_generator: FewShotGenerator | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    logger.info(f"Evaluate the model with gen_kwargs: {gen_kwargs}")

    # Load the evaluation dataset
    eval_instances: Sequence[ChatInstance] = eval_dataset
    if max_instances is not None:
        eval_instances = [eval_dataset[i] for i in range(min(max_instances, len(eval_dataset)))]

    # Generate responses for each instance
    all_messages_list: list[list[dict[str, str]]] = []
    references_list: list[list[str]] = []
    extra_info_list: list[dict[str, Any]] = []
    with tqdm(total=len(eval_instances)) as pbar:
        for batch_id, batch in enumerate(batch_iter(eval_instances, batch_size)):
            input_messages_list = [chat_instance.messages for chat_instance in batch]

            # Generate few-shot instances
            # The few-shot examples here follow a multi-turn format, interleaving user and assistant messages.
            if few_shot_generator is not None:
                for input_id in range(len(input_messages_list)):
                    few_shot_instances = few_shot_generator(eval_inputs=input_messages_list[input_id])
                    few_shot_messages: list[dict[str, str]] = []
                    for few_shot_instance in few_shot_instances:
                        if not isinstance(few_shot_instance, ChatInstance):
                            msg = f"Invalid instance type: {type(few_shot_instance)}"
                            raise TypeError(msg)
                        few_shot_messages += few_shot_instance.messages
                        if few_shot_instance.references:
                            # use the first reference as the assistant message
                            few_shot_messages += [{"role": "assistant", "content": few_shot_instance.references[0]}]
                    input_messages_list[input_id] = [*few_shot_messages, *input_messages_list[input_id]]

            if not eval_dataset.require_incremental_response():
                # Continue generation from the given conversation history
                lm_outputs: list[LMOutput] = language_model.generate_chat_response(
                    input_messages_list,
                    **gen_kwargs,
                )
                for input_messages, lm_output in zip(input_messages_list, lm_outputs):
                    all_messages_list.append(
                        [
                            *input_messages,
                            {
                                "role": "assistant",
                                "content": lm_output.text,
                                "finish_reason": lm_output.finish_reason,
                            }
                            | ({"raw_content": lm_output.raw_text} if lm_output.raw_text else {}),
                        ],
                    )
            else:
                # In incremental response generation,
                # the input messages are supposed to be the user messages across turns.
                # The model first responses to the first user message, then add its response to the chat history,
                # and responses to the next user message, and so on.
                max_num_turns = max(len(messages) for messages in input_messages_list)
                current_chat_history: list[list[dict[str, str]]] = [[] for _ in input_messages_list]
                # perform generation for each turn
                for turn in range(max_num_turns):
                    batch_ids_fed_to_model = [
                        b_id for b_id, messages in enumerate(input_messages_list) if turn < len(messages)
                    ]
                    current_model_inputs = [
                        _remove_redundant_keys_from_messages(
                            current_chat_history[b_id] + [input_messages_list[b_id][turn]],
                            remove_keys={"finish_reason", "raw_content"},
                        )
                        for b_id in batch_ids_fed_to_model
                    ]
                    lm_outputs = language_model.generate_chat_response(
                        current_model_inputs,
                        **gen_kwargs,
                    )
                    for o_id, b_id in enumerate(batch_ids_fed_to_model):
                        current_chat_history[b_id].append(input_messages_list[b_id][turn])
                        current_chat_history[b_id].append(
                            {
                                "role": "assistant",
                                "content": lm_outputs[o_id].text,
                                "finish_reason": lm_outputs[o_id].finish_reason,
                            }
                            | ({"raw_content": lm_outputs[o_id].raw_text} if lm_outputs[o_id].raw_text else {})
                        )
                all_messages_list += current_chat_history

            references_list += [chat_instance.references for chat_instance in batch]
            extra_info_list += [chat_instance.extra_info for chat_instance in batch]

            if batch_id == 0:
                logger.info("Example of the conversation")
                logger.info(f"{all_messages_list[0]}")

            pbar.update(len(batch))

    # Evaluate the generated responses
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

    # Calculate the finish_reason statistics
    finish_reason_counter = Counter()
    for messages in all_messages_list:
        for mes in messages:
            if "finish_reason" in mes:
                finish_reason_counter[mes["finish_reason"]] += 1
    for finish_reason, count in finish_reason_counter.items():
        metrics_summary_dict[f"finish_reason_ratio-{finish_reason}"] = count / sum(finish_reason_counter.values())

    logger.info(metrics_summary_dict)

    outputs = [
        {
            "lm_output": messages[-1]["content"],
            "finish_reason": messages[-1]["finish_reason"],
            "task_inputs": {"messages": messages[:-1], **extra_info},
            "references": references,
            **instance_metrics,
        }
        | ({"raw_lm_output": messages[-1]["raw_content"]} if "raw_content" in messages[-1] else {})
        for messages, references, extra_info, instance_metrics in zip(
            all_messages_list,
            references_list,
            extra_info_list,
            instance_metrics_list,
        )
    ]
    return metrics_summary_dict, outputs
