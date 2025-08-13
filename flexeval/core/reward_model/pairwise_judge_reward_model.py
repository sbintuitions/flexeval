from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from flexeval.core.language_model.base import LanguageModel
from flexeval.core.prompt_template.base import PromptTemplate
from flexeval.core.reward_bench_dataset import RewardBenchInstance
from flexeval.core.reward_model.base import RewardModel


class PairwiseChoice(str, Enum):
    A = "[[A]]"
    B = "[[B]]"


@dataclass
class PairwiseInstance:
    """
    A dataclass representing a pairwise instance for a reward bench task.
    Unlike RewardBenchInstance, this class represents the chosen instance (answer) by answer label,
    in order to make the order insensitive input for the language model.
    """

    prompt: list[dict[str, str]]
    answer_a: list[dict[str, str]]
    answer_b: list[dict[str, str]]
    answer_label: PairwiseChoice


def evaluate_model_output(model_output: str, gold_label: PairwiseChoice) -> bool:
    # If both choices are in model output, then output is **wrong**
    if PairwiseChoice.A.value in model_output and PairwiseChoice.B.value in model_output:
        return False

    # If only gold label is in model output, then output is **correct**
    if gold_label.value in model_output:
        return True
    return False


def aggregate_judge_results(
    outputs: list[dict],
    judge_outputs: list,
    chosen_is_better_list: list[bool],
) -> tuple[list[bool], list[dict]]:
    """
    Aggregates the doubled-length `judge_outputs` and `chosen_is_better_list` into final results for each individual instance.

    Returns:
        final_results: list[bool]
            A list indicating whether each instance is ultimately judged as correct.
    """
    aggregated_results: list[bool] = []
    aggregated_outputs: list[dict] = []

    for i, output in enumerate(outputs):
        ab_output_text = judge_outputs[i * 2].text
        ba_output_text = judge_outputs[i * 2 + 1].text
        ab_eval = chosen_is_better_list[i * 2]
        ba_eval = chosen_is_better_list[i * 2 + 1]

        consistent = ab_eval == ba_eval
        final_is_correct = consistent and ab_eval

        aggregated_outputs.append(
            {
                "llm_outputs": [ab_output_text, ba_output_text],
                "evaluation_results": [ab_eval, ba_eval],
                "consistent": consistent,
                "final_is_correct": final_is_correct,
                **output,
            }
        )
        aggregated_results.append(final_is_correct)

    return aggregated_results, aggregated_outputs


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
            elif isinstance(self.system_message, PromptTemplate):
                system_message = self.system_message.embed_inputs(pairwise_instance_asdict)
            else:
                msg = "system_message should be str or PromptTemplate."
                raise ValueError(msg)
            input_chat_messages.insert(
                0,
                {"role": "system", "content": system_message},
            )
        return input_chat_messages

    def batch_judge(
        self,
        batch_reward_bench_instances: list[RewardBenchInstance],
    ) -> tuple[list[bool], list[dict[str, Any]]]:
        input_chat_messages_list: list[list[dict[str, str]]] = []
        all_pairwise_instances: list[PairwiseInstance] = []
        outputs: list[dict[str, Any]] = []
        for reward_bench_instance in batch_reward_bench_instances:
            # to address position biases, create two inputs by swapping chosen/rejected orderings
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
        judge_outputs = self.language_model.generate_chat_response(input_chat_messages_list, **self.gen_kwargs)
        chosen_is_better_list: list[bool] = [
            evaluate_model_output(judge_output.text, pairwise_instance.answer_label)
            for judge_output, pairwise_instance in zip(judge_outputs, all_pairwise_instances)
        ]

        if len(outputs) * 2 != len(chosen_is_better_list):
            msg = "The number of outputs should be twice the number of inputs."
            raise ValueError(msg)

        aggregated_results, aggregated_outputs = aggregate_judge_results(outputs, judge_outputs, chosen_is_better_list)

        return aggregated_results, aggregated_outputs
