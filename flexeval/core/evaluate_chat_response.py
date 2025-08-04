from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from loguru import logger
from tqdm import tqdm

from flexeval.core.language_model.base import LMOutput
from flexeval.core.utils.chat_util import find_first_turn_for_response

from .chat_dataset import ChatDataset, ChatInstance
from .few_shot_generator import FewShotGenerator
from .language_model import LanguageModel
from .metric import Metric
from .utils.data_util import batch_iter


def _remove_redundant_keys_from_messages(
    messages: list[dict[str, Any]],
    remove_keys: Iterable[str],
) -> list[dict[str, Any]]:
    """
    Remove specified keys from all turns in `messages`.

    Some keys such as `finish_reason` are added in `evaluate_chat_response()` for logging.
    However, some APIs of Azure OpenAI (e.g. tsuzumi-7b-instruct) do not allow extra input keys.
    Thus this removal is required before each input step.
    """
    remove_keys = set(remove_keys)
    return [{key: value for key, value in message.items() if key not in remove_keys} for message in messages]


def evaluate_chat_response(  # noqa: C901,PLR0912, PLR0915
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
    all_messages_list: list[list[dict[str, Any]]] = []
    references_list: list[list[str]] = []
    extra_info_list: list[dict[str, Any]] = []
    all_input_tools_list: list[list[dict[str, Any]] | None] = []
    with tqdm(total=len(eval_instances)) as pbar:
        for batch_id, batch in enumerate(batch_iter(eval_instances, batch_size)):
            input_messages_list = [chat_instance.messages for chat_instance in batch]
            input_tools_list = [chat_instance.tools for chat_instance in batch]
            all_input_tools_list += input_tools_list
            if all(tools is None for tools in input_tools_list):
                input_tools_list = None

            # For the `require_incremental_response==True` case,
            # it is necessary to identify the first turn that should be responded, excluding system messages, etc.
            offsets_to_first_turn = [find_first_turn_for_response(messages) for messages in input_messages_list]

            # Generate few-shot instances
            # The few-shot examples here follow a multi-turn format, interleaving user and assistant messages.
            if few_shot_generator is not None:
                for input_id in range(len(input_messages_list)):
                    few_shot_instances = few_shot_generator(eval_inputs=input_messages_list[input_id])
                    few_shot_messages: list[dict[str, Any]] = []
                    for few_shot_instance in few_shot_instances:
                        if not isinstance(few_shot_instance, ChatInstance):
                            msg = f"Invalid instance type: {type(few_shot_instance)}"
                            raise TypeError(msg)
                        few_shot_messages += few_shot_instance.messages
                        if few_shot_instance.references:
                            # use the first reference as the assistant message
                            few_shot_messages += [{"role": "assistant", "content": few_shot_instance.references[0]}]

                    offset = offsets_to_first_turn[input_id]
                    meta_messages = input_messages_list[input_id][:offset]
                    real_messages = input_messages_list[input_id][offset:]
                    input_messages_list[input_id] = [*meta_messages, *few_shot_messages, *real_messages]
                    offsets_to_first_turn[input_id] += len(few_shot_messages)

            if not eval_dataset.require_incremental_response():
                # Continue generation from the given conversation history
                lm_outputs: list[LMOutput] = language_model.generate_chat_response(
                    input_messages_list,
                    tools=input_tools_list,
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
                            | ({"raw_content": lm_output.raw_text} if lm_output.raw_text else {})
                            | ({"tool_calls": lm_output.tool_calls} if lm_output.tool_calls else {})
                            | (
                                {"tool_call_validation_result": lm_output.tool_call_validation_result}
                                if lm_output.tool_call_validation_result
                                else {}
                            ),
                        ],
                    )
            else:
                # In incremental response generation,
                # the input messages are supposed to be the user messages across turns.
                # The model first responses to the first user message, then add its response to the chat history,
                # and responses to the next user message, and so on.
                max_num_turns = max(
                    len(messages) - offset for messages, offset in zip(input_messages_list, offsets_to_first_turn)
                )
                current_chat_history: list[list[dict[str, Any]]] = [
                    input_messages[:offset]
                    for input_messages, offset in zip(input_messages_list, offsets_to_first_turn)
                ]
                # perform generation for each turn
                for turn in range(max_num_turns):
                    batch_ids_fed_to_model = [
                        b_id
                        for b_id, messages in enumerate(input_messages_list)
                        if turn < (len(messages) - offsets_to_first_turn[b_id])
                    ]
                    current_model_inputs = [
                        _remove_redundant_keys_from_messages(
                            current_chat_history[b_id]
                            + [input_messages_list[b_id][turn + offsets_to_first_turn[b_id]]],
                            remove_keys={"finish_reason", "raw_content", "tool_call_validation_result"},
                        )
                        for b_id in batch_ids_fed_to_model
                    ]
                    lm_outputs = language_model.generate_chat_response(
                        current_model_inputs,
                        tools=[input_tools_list[b_id] for b_id in batch_ids_fed_to_model] if input_tools_list else None,
                        **gen_kwargs,
                    )
                    for o_id, b_id in enumerate(batch_ids_fed_to_model):
                        offset = offsets_to_first_turn[b_id]
                        current_chat_history[b_id].append(input_messages_list[b_id][turn + offset])
                        current_chat_history[b_id].append(
                            {
                                "role": "assistant",
                                "content": lm_outputs[o_id].text,
                                "finish_reason": lm_outputs[o_id].finish_reason,
                            }
                            | ({"raw_content": lm_outputs[o_id].raw_text} if lm_outputs[o_id].raw_text else {})
                            | ({"tool_calls": lm_outputs[o_id].tool_calls} if lm_outputs[o_id].tool_calls else {})
                            | (
                                {"tool_call_validation_result": lm_outputs[o_id].tool_call_validation_result}
                                if lm_outputs[o_id].tool_call_validation_result
                                else {}
                            ),
                        )
                all_messages_list += current_chat_history

            references_list += [chat_instance.references for chat_instance in batch]
            extra_info_list += [chat_instance.extra_info for chat_instance in batch]

            if batch_id == 0:
                logger.info("Example of the conversation")
                logger.info(f"{all_messages_list[0]}")

            pbar.update(len(batch))

    for extra_info, messages, tools in zip(extra_info_list, all_messages_list, all_input_tools_list):
        last_message = messages[-1]
        if "tool_calls" in last_message:
            extra_info["tool_calls"] = last_message["tool_calls"]
        if "tool_call_validation_result" in last_message:
            extra_info["tool_call_validation_result"] = last_message["tool_call_validation_result"]
        if tools:
            extra_info["tools"] = tools

    # Metric.evaluate() accepts only str, so if content is None, convert it to empty string
    for messages in all_messages_list:
        for mes in messages:
            if mes["content"] is None:
                mes["content"] = ""

    language_model.cleanup_resources()

    # Evaluate the generated responses
    metrics_summary_dict: dict[str, float] = {}
    instance_metrics_list: list[dict[str, Any]] = [{} for _ in range(len(all_messages_list))]
    for metric in metrics:
        metric_result = metric.evaluate(
            lm_outputs=[messages[-1]["content"] for messages in all_messages_list],
            references_list=references_list,
            extra_info_list=[
                {"messages": messages[:-1], **extra_info}
                for messages, extra_info in zip(all_messages_list, extra_info_list)
            ],
        )
        metric.cleanup_resources()

        metrics_summary_dict.update(metric_result.summary)

        if metric_result.instance_details:
            for instance_idx, instance_details in enumerate(
                metric_result.instance_details,
            ):
                instance_metrics_list[instance_idx].update(instance_details)

    # Calculate the finish_reason and validation statistics
    finish_reason_counter = Counter()
    tool_call_validation_result_counter = Counter()
    for messages in all_messages_list:
        for mes in messages:
            if "finish_reason" in mes:
                finish_reason_counter[mes["finish_reason"]] += 1
            if "tool_call_validation_result" in mes:
                tool_call_validation_result_counter[mes["tool_call_validation_result"]] += 1
    for finish_reason, count in finish_reason_counter.items():
        metrics_summary_dict[f"finish_reason_ratio-{finish_reason}"] = count / sum(finish_reason_counter.values())
    for validation_result, count in tool_call_validation_result_counter.items():
        metrics_summary_dict[f"tool_call_validation_result_ratio-{validation_result}"] = count / sum(
            tool_call_validation_result_counter.values()
        )

    logger.info(metrics_summary_dict)

    outputs = [
        {
            "lm_output": messages[-1]["content"],
            "finish_reason": messages[-1]["finish_reason"],
            "extra_info": {"messages": messages[:-1], **extra_info},
            "references": references,
            **instance_metrics,
        }
        | ({"raw_lm_output": messages[-1]["raw_content"]} if "raw_content" in messages[-1] else {})
        | (
            {"tool_call_validation_result": messages[-1]["tool_call_validation_result"]}
            if "tool_call_validation_result" in messages[-1]
            else {}
        )
        for messages, references, extra_info, instance_metrics in zip(
            all_messages_list,
            references_list,
            extra_info_list,
            instance_metrics_list,
        )
    ]
    return metrics_summary_dict, outputs
