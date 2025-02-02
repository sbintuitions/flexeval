from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from flexeval.core.reward_bench_dataset import RewardBenchInstance
from flexeval.core.reward_model.base import RewardModel
from flexeval.utils.hf_utils import get_default_model_kwargs


class SequenceClassificationRewardModel(RewardModel):
    """Pairwise judge using a chat language model to compare two model or human
    outputs.
    """

    def __init__(
        self,
        model: str,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        tokenizer = tokenizer if tokenizer else model
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)

        model_kwargs = get_default_model_kwargs(model_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, **model_kwargs)
        # Set pad_token_id if not set
        # to avoid "ValueError: Cannot handle batch sizes > 1 if no padding token is defined." in self.model()
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    @torch.inference_mode()
    def batch_judge(
        self,
        batch_reward_bench_instances: list[RewardBenchInstance],
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        chosen_messages = [instance.prompt + instance.chosen for instance in batch_reward_bench_instances]
        chosen_inputs = self.tokenizer.apply_chat_template(
            chosen_messages, return_tensors="pt", padding=True, return_dict=True
        )
        chosen_outputs = self.model(**{k: v.to(self.model.device) for k, v in chosen_inputs.items()})
        chosen_rewards = chosen_outputs.logits[:, 0]

        rejected_messages = [instance.prompt + instance.rejected for instance in batch_reward_bench_instances]
        rejected_inputs = self.tokenizer.apply_chat_template(
            rejected_messages, return_tensors="pt", padding=True, return_dict=True
        )
        rejected_outputs = self.model(**{k: v.to(self.model.device) for k, v in rejected_inputs.items()})
        rejected_rewards = rejected_outputs.logits[:, 0]

        chosen_is_better = (chosen_rewards > rejected_rewards).tolist()
        outputs = [
            {
                "chosen_reward": chosen_reward.item(),
                "rejected_reward": rejected_reward.item(),
            }
            for chosen_reward, rejected_reward in zip(chosen_rewards, rejected_rewards)
        ]
        return chosen_is_better, outputs
