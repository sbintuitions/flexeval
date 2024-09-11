from __future__ import annotations

from dataclasses import asdict
from typing import Any

from flexeval.core.language_model.base import LanguageModel
from flexeval.core.prompt_template.base import PromptTemplate
from flexeval.core.reward_bench_dataset.reward_bench_dataset import (
    RewardBenchInstance,
    ShufflePairwiseInstance,
    reward_bench_instance_to_shuffle_pairwise_instance,
)
from flexeval.core.reward_model.base import PairwiseJudgeUsingRewardModel


class PairwiseJudgeUsingRewardLLM(PairwiseJudgeUsingRewardModel):
    """Pairwise judge using a chat language model to compare two model or human
    outputs.

    Args:
        language_model: The language model to use for pairwise comparison.
        prompt_template: The prompt template to embed the model outputs to be compared.
        system_message: The system message to prepend to the chat messages.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        system_message: str | PromptTemplate | None = None,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.system_message = system_message

    def _is_correct(self,
                    judge_output: str,
                    pairwise_instances: ShufflePairwiseInstance) -> bool:
        if judge_output == pairwise_instances.chosen:
            return True
        return False

    def batch_judge(
        self,
        batch_reward_bench_instances: list[RewardBenchInstance],
        gen_kwargs: dict[str, Any],
    ) -> tuple[list[str], list[bool]]:
        input_chat_messages_list: list[list[dict[str, str]]] = []
        shuffle_pairwise_instances: list[ShufflePairwiseInstance] = [
            reward_bench_instance_to_shuffle_pairwise_instance(batch_reward_bench_instance)
            for batch_reward_bench_instance in batch_reward_bench_instances
        ]
        for shuffle_pairwise_instance in shuffle_pairwise_instances:
            judge_input = self.prompt_template.embed_inputs(asdict(shuffle_pairwise_instance))
            input_chat_messages = [{"role": "user", "content": judge_input}]
            if self.system_message:
                if isinstance(self.system_message, str):
                    system_message = self.system_message
                else:
                    system_message = self.system_message.embed_inputs(shuffle_pairwise_instance)
                input_chat_messages.insert(
                    0,
                    {"role": "system", "content": system_message},
                )
            input_chat_messages_list.append(input_chat_messages)
        judge_outputs = self.language_model.batch_generate_chat_response(input_chat_messages_list, **gen_kwargs)
        judge_results = [
            self._is_correct(judge_output, shuffle_pairwise_instance)
            for judge_output, shuffle_pairwise_instance in zip(judge_outputs, shuffle_pairwise_instances)
        ]
        return judge_outputs, judge_results
