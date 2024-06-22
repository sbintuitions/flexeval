from __future__ import annotations

from typing import Any, Sequence

from loguru import logger
from tqdm import tqdm

from .few_shot_generator import FewShotGenerator
from .generation_dataset import GenerationDataset, GenerationInstance
from .language_model import LanguageModel
from .metric import Metric
from .prompt_template import PromptTemplate
from .utils.data_util import batch_iter


def evaluate_generation(  # noqa: C901
    language_model: LanguageModel,
    gen_kwargs: dict[str, Any],
    eval_dataset: GenerationDataset,
    prompt_template: PromptTemplate,
    metrics: list[Metric],
    batch_size: int,
    max_instances: int | None = None,
    few_shot_generator: FewShotGenerator | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    logger.info(f"Evaluate the model with gen_kwargs: {gen_kwargs}")
    logger.info(f"Prompt template: {prompt_template}")

    eval_instances: Sequence[GenerationInstance] = eval_dataset
    if max_instances is not None:
        eval_instances = [eval_dataset[i] for i in range(min(max_instances, len(eval_dataset)))]

    lm_prompt_list: list[str] = []
    lm_output_list: list[str] = []
    with tqdm(total=len(eval_instances)) as pbar:
        for i, batch in enumerate(batch_iter(eval_instances, batch_size)):
            lm_prompts: list[str] = []
            for eval_instance in batch:
                template_inputs = eval_instance.inputs
                if few_shot_generator is not None:
                    few_shot_instances = few_shot_generator(template_inputs)
                    few_shot_item_list: list[dict[str, Any]] = []
                    for few_shot_instance in few_shot_instances:
                        if isinstance(few_shot_instance, GenerationInstance):
                            few_shot_item = {**few_shot_instance.inputs, "references": few_shot_instance.references}
                            few_shot_item_list.append(few_shot_item)
                        else:
                            msg = f"Invalid instance type: {type(few_shot_instance)}"
                            raise TypeError(msg)
                    template_inputs = {**template_inputs, "few_shot_data": few_shot_item_list}
                prompt = prompt_template.embed_inputs(template_inputs)
                lm_prompts.append(prompt)

            lm_outputs = language_model.batch_complete_text(
                lm_prompts,
                **gen_kwargs,
            )

            if i == 0:
                logger.info("Example of the model inputs and outputs:")
                logger.info(f"lm_prompts: {lm_prompts[0]}")
                logger.info(f"lm_outputs: {lm_outputs[0]}")

            lm_prompt_list += lm_prompts
            lm_output_list += lm_outputs

            pbar.update(len(batch))
    metrics_summary_dict: dict[str, float] = {}
    instance_metrics_list: list[dict[str, Any]] = [{} for _ in range(len(eval_instances))]
    for metric in metrics:
        metric_result = metric.evaluate(
            lm_outputs=lm_output_list,
            references_list=[i.references for i in eval_instances],
            task_inputs_list=[i.inputs for i in eval_instances],
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
            "lm_prompt": lm_prompt,
            "lm_output": lm_output,
            "task_inputs": eval_instance.inputs,
            "references": eval_instance.references,
            **instance_metrics,
        }
        for lm_prompt, lm_output, eval_instance, instance_metrics in zip(
            lm_prompt_list,
            lm_output_list,
            eval_instances,
            instance_metrics_list,
        )
    ]
    return metrics_summary_dict, outputs
