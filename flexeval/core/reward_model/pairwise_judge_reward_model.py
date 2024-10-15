from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from flexeval.core.language_model.base import LanguageModel
from flexeval.core.prompt_template.base import PromptTemplate
from flexeval.core.reward_bench_dataset.hf import RewardBenchInstance
from flexeval.core.reward_model.base import RewardModel


class PairwiseChoice(str, Enum):
    A = "[[A]]"
    B = "[[B]]"


@dataclass
class PairwiseInstance:
    prompt: str
    answer_a: str
    answer_b: str
    answer_label: PairwiseChoice


class PairwiseJudgeRewardModel(RewardModel):
    """Pairwise judge using a chat language model to compare two model or human
    outputs.

    Args:
        language_model: The language model to use for pairwise comparison.
                        This model is expected to output PairwiseChoice.
        prompt_template: The prompt template to embed the model outputs to be compared.
                         Be sure to include {{prompt}}, {{answer_a}}, and {{answer_b}}.
        system_message: The system message to prepend to the chat messages.
        gen_kwargs: Generation kwargs for the language model.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        system_message: str | PromptTemplate | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if gen_kwargs is None:
            gen_kwargs = {}
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.system_message = system_message
        self.gen_kwargs = gen_kwargs

    def _create_input_chat_messages_list(self, pairwise_instance: PairwiseInstance) -> list[dict[str, str]]:
        pairwise_instance_asdict = asdict(pairwise_instance)
        judge_input = self.prompt_template.embed_inputs(pairwise_instance_asdict)
        input_chat_messages = [{"role": "user", "content": judge_input}]
        if self.system_message:
            if isinstance(self.system_message, str):
                system_message = self.system_message
            else:
                system_message = self.system_message.embed_inputs(pairwise_instance_asdict)
            input_chat_messages.insert(
                0,
                {"role": "system", "content": system_message},
            )
        return input_chat_messages

    def _is_correct_llm_answer(self, llm_answer: str, pairwise_choice: PairwiseChoice) -> bool:
        # Check if the answer is one of the valid choices.
        if PairwiseChoice.A.value in PairwiseChoice.B in llm_answer and PairwiseChoice.B in llm_answer:
            return False
        if pairwise_choice.value in llm_answer:
            return True
        return False

    def batch_judge(
        self,
        batch_reward_bench_instances: list[RewardBenchInstance],
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        input_chat_messages_list: list[list[dict[str, str]]] = []
        all_pairwise_instances: list[PairwiseInstance] = []
        outputs: list[dict[str, Any]] = []
        for reward_bench_instance in batch_reward_bench_instances:
            pairwise_instance_answer_a_is_chosen = PairwiseInstance(
                prompt=reward_bench_instance.prompt,
                answer_a=reward_bench_instance.chosen,
                answer_b=reward_bench_instance.rejected,
                answer_label=PairwiseChoice.A,
            )
            input_chat_messages_a_is_chosen = self._create_input_chat_messages_list(
                pairwise_instance_answer_a_is_chosen
            )
            input_chat_messages_list.append(input_chat_messages_a_is_chosen)

            pairwise_instance_answer_b_is_chosen = PairwiseInstance(
                prompt=reward_bench_instance.prompt,
                answer_a=reward_bench_instance.rejected,
                answer_b=reward_bench_instance.chosen,
                answer_label=PairwiseChoice.B,
            )
            input_chat_messages_b_is_chosen = self._create_input_chat_messages_list(
                pairwise_instance_answer_b_is_chosen
            )
            input_chat_messages_list.append(input_chat_messages_b_is_chosen)
            all_pairwise_instances += [pairwise_instance_answer_a_is_chosen, pairwise_instance_answer_b_is_chosen]

            output = {
                "llm_inputs": [input_chat_messages_a_is_chosen, input_chat_messages_b_is_chosen],
            }
            outputs.append(output)
        judge_outputs = self.language_model.batch_generate_chat_response(input_chat_messages_list, **self.gen_kwargs)
        chosen_is_betters: list[bool] = [
            self._is_correct_llm_answer(judge_output, shuffle_pairwise_instance.answer_label)
            for judge_output, shuffle_pairwise_instance in zip(judge_outputs, all_pairwise_instances)
        ]

        if len(outputs) * 2 != len(chosen_is_betters):
            msg = "The number of outputs should be twice the number of inputs."
            raise ValueError(msg)

        for i in range(len(outputs)):
            outputs[i]["llm_outputs"] = [judge_outputs[i * 2], judge_outputs[i * 2 + 1]]
            outputs[i]["evaluation_results"] = [chosen_is_betters[i * 2], chosen_is_betters[i * 2 + 1]]

        return chosen_is_betters, outputs
