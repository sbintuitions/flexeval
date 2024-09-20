from __future__ import annotations

from typing import Any, Sequence

from loguru import logger
from tqdm import tqdm

from flexeval.core.reward_bench_dataset.hf import RewardBenchDataset, RewardBenchInstance
from flexeval.core.reward_model.base import RewardModel
from flexeval.core.utils.data_util import batch_iter


def evaluate_reward_model(
    reward_model: RewardModel,
    eval_dataset: RewardBenchDataset,
    batch_size: int,
    max_instances: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    reward_bench_instances: Sequence[RewardBenchInstance] = eval_dataset
    if max_instances is not None:
        reward_bench_instances = [eval_dataset[i] for i in range(min(max_instances, len(eval_dataset)))]

    judge_outputs: list[str] = []
    judge_results: list[bool] = []
    with tqdm(total=len(reward_bench_instances)) as pbar:
        for i, batch_reward_bench_instances in enumerate(batch_iter(reward_bench_instances, batch_size)):
            judge_outputs_i, judge_results_i = reward_model.batch_judge(batch_reward_bench_instances)
            judge_outputs += judge_outputs_i
            judge_results += judge_results_i

            if i == 0:
                logger.info("Example of the model inputs and outputs:")
                logger.info(f"prompt: {batch_reward_bench_instances[0].prompt}")
                logger.info(f"chosen: {batch_reward_bench_instances[0].chosen}")
                logger.info(f"rejected: {batch_reward_bench_instances[0].rejected}")
                logger.info(f"Output: {judge_outputs_i[0]}")

            pbar.update(len(batch_reward_bench_instances))

    accuracy = sum(judge_results) / len(judge_results)
    return {"accuracy": accuracy}, judge_outputs
