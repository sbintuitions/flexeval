from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Iterator, Sequence

from loguru import logger
from tqdm import tqdm

from flexeval.core.utils.chat_util import find_first_turn_for_response

from .chat_dataset import ChatInstance
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


def _add_few_shot_messages_to_chat_instance(chat_instance: ChatInstance, few_shot_generator: FewShotGenerator) -> None:
    """Add few-shot examples to a chat instance by inserting them before the first user message.
    Note that it updates ChatInstance in-place.
    """
    few_shot_instances = few_shot_generator(eval_inputs=chat_instance.messages)
    if not all(isinstance(inst, ChatInstance) for inst in few_shot_instances):
        msg = "few_shot_instances must only contain ChatInstance, but it does not"
        raise TypeError(msg)

    few_shot_messages: list[dict[str, Any]] = []
    for few_shot_instance in few_shot_instances:
        few_shot_messages += few_shot_instance.messages
        if few_shot_instance.references:
            # use the first reference as the assistant message
            few_shot_messages += [{"role": "assistant", "content": few_shot_instance.references[0]}]

    # Insert few_shot_messages before the first user message
    first_turn_idx = find_first_turn_for_response(chat_instance.messages)
    chat_instance.messages = [
        *chat_instance.messages[:first_turn_idx],
        *few_shot_messages,
        *chat_instance.messages[first_turn_idx:],
    ]


def _find_response_context_index(incomplete_messages: list[dict[str, Any]]) -> int | None:
    """
    Finds the index after the earliest message that requires an assistant response.

    Specifically, it looks for the first message from a role that expects an assistant reply
    ('user' or 'tool') that is not followed by an 'assistant' message.

    Returns:
        The index to slice messages up to (exclusive of that index), or None if no reply is needed.
        Use as: incomplete_messages[:index]
    """
    expected_roles = {"user", "tool"}

    for i in range(len(incomplete_messages) - 1):
        current = incomplete_messages[i]
        next_msg = incomplete_messages[i + 1]
        if current["role"] in expected_roles and next_msg["role"] != "assistant":
            return i + 1

    # Check if the last message expects a response
    if incomplete_messages and incomplete_messages[-1]["role"] in expected_roles:
        return len(incomplete_messages)

    return None


def execute_conversation_flow(
    language_model: LanguageModel, eval_instances: Sequence[ChatInstance], batch_size: int, gen_kwargs: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """
    Execute complete conversation flows for a batch of chat instances.

    Processes multi-turn conversations by iteratively finding response points,
    generating language model responses, and building complete conversation histories.
    Handles tool calls and continues conversations until all instances are complete.
    """
    for batch in batch_iter(eval_instances, batch_size):
        # Copy the input messages as current_chat_history
        # We will populate this history with llm
        current_chat_history: list[list[dict[str, Any]]] = [[*chat_instance.messages] for chat_instance in batch]
        response_context_indices = [_find_response_context_index(chat_history) for chat_history in current_chat_history]
        while any(idx is not None for idx in response_context_indices):
            batch_inputs = [
                _remove_redundant_keys_from_messages(
                    current_chat_history[batch_i][:message_i],
                    remove_keys={"finish_reason", "raw_content", "tool_call_validation_result"},
                )
                for batch_i, message_i in enumerate(response_context_indices)
                if message_i is not None
            ]
            ids_fed_to_lm = [i for i, idx in enumerate(response_context_indices) if idx is not None]
            lm_outputs = language_model.generate_chat_response(
                batch_inputs,
                tools=[batch[i].tools for i in ids_fed_to_lm],
                **gen_kwargs,
            )
            for lm_output, batch_i in zip(lm_outputs, ids_fed_to_lm):
                lm_output_message = (
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
                    )
                )
                current_chat_history[batch_i].insert(response_context_indices[batch_i], lm_output_message)
            response_context_indices = [
                _find_response_context_index(chat_history) for chat_history in current_chat_history
            ]

        for messages, chat_instance in zip(current_chat_history, batch):
            extra_info = chat_instance.extra_info
            last_message = messages[-1]
            if "tool_calls" in last_message:
                extra_info["tool_calls"] = last_message["tool_calls"]
            if "tool_call_validation_result" in last_message:
                extra_info["tool_call_validation_result"] = last_message["tool_call_validation_result"]
            if chat_instance.tools:
                extra_info["tools"] = chat_instance.tools
            output = {
                "lm_output": messages[-1]["content"],
                "finish_reason": messages[-1]["finish_reason"],
                "extra_info": {"messages": messages, **extra_info},
                "references": chat_instance.references,
            }
            if "raw_content" in messages[-1]:
                output["raw_lm_output"] = messages[-1]["raw_content"]
            if "tool_call_validation_result" in messages[-1]:
                output["tool_call_validation_result"] = messages[-1]["tool_call_validation_result"]
            yield output


def evaluate_chat_response(  # noqa: C901, PLR0912
    language_model: LanguageModel,
    gen_kwargs: dict[str, Any],
    eval_dataset: Sequence[ChatInstance],
    metrics: list[Metric],
    batch_size: int,
    few_shot_generator: FewShotGenerator | None = None,
    max_instances: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    logger.info(f"Evaluate the model with gen_kwargs: {gen_kwargs}")

    eval_instances: Sequence[ChatInstance] = eval_dataset
    if max_instances is not None:
        eval_instances = [eval_dataset[i] for i in range(min(max_instances, len(eval_instances)))]

    if few_shot_generator:
        for eval_instance in eval_instances:
            _add_few_shot_messages_to_chat_instance(eval_instance, few_shot_generator)

    # Generate responses for each instance
    outputs: list[dict[str, Any]] = []
    for i, output in enumerate(
        tqdm(
            execute_conversation_flow(
                language_model=language_model,
                eval_instances=eval_instances,
                batch_size=batch_size,
                gen_kwargs=gen_kwargs,
            ),
            total=len(eval_instances),
        )
    ):
        if i == 0:
            logger.info("Example of the output")
            logger.info(json.dumps(output, ensure_ascii=False, indent=4))
        outputs.append(output)

    language_model.cleanup_resources()

    # Evaluate the generated responses
    metrics_summary_dict: dict[str, float] = {}
    instance_metrics_list: list[dict[str, Any]] = [{} for _ in range(len(outputs))]
    for metric in metrics:
        metric_result = metric.evaluate(
            lm_outputs=[output["lm_output"] for output in outputs],
            references_list=[output["references"] for output in outputs],
            extra_info_list=[output["extra_info"] for output in outputs],
        )
        metric.cleanup_resources()

        metrics_summary_dict.update(metric_result.summary)

        if metric_result.instance_details:
            for instance_idx, instance_details in enumerate(
                metric_result.instance_details,
            ):
                instance_metrics_list[instance_idx].update(instance_details)
    for instance_metrics, output in zip(instance_metrics_list, outputs):
        output.update(instance_metrics)

    # Calculate the finish_reason and validation statistics
    finish_reason_counter = Counter()
    tool_call_validation_result_counter = Counter()
    for output in outputs:
        messages = output["extra_info"]["messages"]
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
    return metrics_summary_dict, outputs
