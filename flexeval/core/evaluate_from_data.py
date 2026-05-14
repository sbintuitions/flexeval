from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from loguru import logger

from flexeval.core.language_model.base import LMOutput

from .chat_dataset import ChatDataset
from .generation_dataset import GenerationDataset
from .metric import Metric


def evaluate_from_data(  # noqa: C901
    eval_data: Iterable[dict[str, Any]],
    metrics: list[Metric],
    eval_dataset: GenerationDataset | ChatDataset | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    extra_info_list: list[dict[str, Any]] = []
    lm_output_list: list[str | LMOutput] = []
    references_list: list[list[str]] = []
    for item in eval_data:
        # ignore extra info if empty for backward compatibility
        # it is OK with some metrics that do not require extra info
        if "extra_info" not in item:
            item["extra_info"] = {}
        extra_info_list.append(item["extra_info"])
        # We adapt the current output format of evaluate_chat_response(). This will be changed in the future.
        if isinstance(item["lm_output"], str):
            lm_output = LMOutput(
                text=item["lm_output"],
                raw_text=item.get("raw_lm_output"),
                reasoning_text=item.get("reasoning_text"),
                finish_reason=item.get("finish_reason"),
                tool_calls=item.get("tool_calls"),
                tool_call_validation_result=item.get("tool_call_validation_result"),
            )
        elif isinstance(item["lm_output"], LMOutput):
            lm_output = item["lm_output"]
        else:
            msg = f"Invalid type for lm_output: {type(item['lm_output'])}"
            raise TypeError(msg)

        lm_output_list.append(lm_output)
        references_list.append(item["references"])

    if eval_dataset is not None:
        if len(eval_dataset) != len(extra_info_list):
            msg = (
                f"The number of instances in the generation_dataset ({len(eval_dataset)}) "
                f"and the eval_file ({len(extra_info_list)}) do not match."
            )
            raise ValueError(msg)
        # override the references_list with the data from the generation_dataset
        for i, eval_instance in enumerate(eval_dataset):
            references_list[i] = eval_instance.references

    metrics_summary_dict: dict[str, float] = {}
    instance_metrics_list: list[dict[str, Any]] = [{} for _ in range(len(extra_info_list))]
    for metric in metrics:
        metric_result = metric.evaluate(
            lm_outputs=lm_output_list,
            references_list=references_list,
            extra_info_list=extra_info_list,
        )
        metric.cleanup_resources()

        metrics_summary_dict.update(metric_result.summary)

        if metric_result.instance_details:
            for instance_idx, instance_details in enumerate(
                metric_result.instance_details,
            ):
                instance_metrics_list[instance_idx].update(instance_details)

    logger.info(metrics_summary_dict)
    return metrics_summary_dict, instance_metrics_list
