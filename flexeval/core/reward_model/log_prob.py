from __future__ import annotations

from typing import Any

from flexeval.core.language_model.base import LanguageModel
from flexeval.core.reward_bench_dataset import RewardBenchInstance
from flexeval.core.reward_model.base import RewardModel


class LogProbRewardModel(RewardModel):
    """
    A reward model that judges the quality of a response
    based on the log probability computed by the auto-regressive language model.
    """

    def __init__(self, language_model: LanguageModel) -> None:
        self.language_model = language_model

    def batch_judge(
        self,
        batch_reward_bench_instances: list[RewardBenchInstance],
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        if not all(len(instance.chosen) == 1 for instance in batch_reward_bench_instances):
            msg = "`chosen` field must have exactly one element."
            raise ValueError(msg)
        if not all(len(instance.rejected) == 1 for instance in batch_reward_bench_instances):
            msg = "`rejected` field must have exactly one element."
            raise ValueError(msg)

        chosen_log_probs = self.language_model.batch_compute_chat_log_probs(
            prompt_list=[instance.prompt for instance in batch_reward_bench_instances],
            response_list=[instance.chosen[0] for instance in batch_reward_bench_instances],
        )
        rejected_log_probs = self.language_model.batch_compute_chat_log_probs(
            prompt_list=[instance.prompt for instance in batch_reward_bench_instances],
            response_list=[instance.rejected[0] for instance in batch_reward_bench_instances],
        )
        chosen_is_better = [
            chosen_log_prob > rejected_log_prob
            for chosen_log_prob, rejected_log_prob in zip(chosen_log_probs, rejected_log_probs)
        ]
        outputs = [
            {
                "chosen_log_prob": chosen_log_prob,
                "rejected_log_prob": rejected_log_prob,
            }
            for chosen_log_prob, rejected_log_prob in zip(chosen_log_probs, rejected_log_probs)
        ]
        return chosen_is_better, outputs
