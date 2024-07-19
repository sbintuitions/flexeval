from __future__ import annotations

from typing import Any, Iterable

from loguru import logger

from .chat_dataset import ChatDataset
from .generation_dataset import GenerationDataset
from .metric import Metric


def evaluate_from_data(
    eval_data: Iterable[dict[str, Any]],
    metrics: list[Metric],
    eval_dataset: GenerationDataset | ChatDataset | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    task_inputs_list: list[dict[str, Any]] = []
    lm_output_list: list[str] = []
    references_list: list[list[str]] = []
    for item in eval_data:
        # ignore task inputs if empty for backward compatibility
        # it is OK with some metrics that do not require task inputs
        if "task_inputs" not in item:
            item["task_inputs"] = {}
        task_inputs_list.append(item["task_inputs"])
        lm_output_list.append(item["lm_output"])
        references_list.append(item["references"])

    if eval_dataset is not None:
        if len(eval_dataset) != len(task_inputs_list):
            msg = (
                f"The number of instances in the generation_dataset ({len(eval_dataset)}) "
                f"and the eval_file ({len(task_inputs_list)}) do not match."
            )
            raise ValueError(msg)
        # override the references_list with the data from the generation_dataset
        for i, eval_instance in enumerate(eval_dataset):
            references_list[i] = eval_instance.references

    metrics_summary_dict: dict[str, float] = {}
    instance_metrics_list: list[dict[str, Any]] = [{} for _ in range(len(task_inputs_list))]
    for metric in metrics:
        metric_result = metric.evaluate(
            lm_outputs=lm_output_list,
            references_list=references_list,
            task_inputs_list=task_inputs_list,
        )

        metrics_summary_dict.update(metric_result.summary)

        if metric_result.instance_details:
            for instance_idx, instance_details in enumerate(
                metric_result.instance_details,
            ):
                instance_metrics_list[instance_idx].update(instance_details)

    logger.info(metrics_summary_dict)
    return metrics_summary_dict, instance_metrics_list
