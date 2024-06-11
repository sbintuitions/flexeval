from __future__ import annotations

import re
from typing import Optional, Union

from flexeval.core.language_model import LanguageModel
from flexeval.core.prompt_template import PromptTemplate
from flexeval.core.utils.data_util import batch_iter

from .base import Metric, MetricResult


class LLMScore(Metric):
    """Let LanguageModel to evaluate the output of another LanguageModel.

    You can specify the evaluation criteria in `PromptTemplate`.
    The last integer value in the output of the evaluator is used as the evaluation score.

    Args:
        language_model: An instance of `LanguageModel` to evaluate the output of the model.
        prompt_template: An instance of `PromptTemplate` to embed the input for the evaluator.
        batch_size: The batch size for the evaluator.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        batch_size: int = 4,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._batch_size = batch_size

    @staticmethod
    def _parse_evaluator_output(evaluator_output: str) -> int:
        """Extract the last integer value from the evaluator output.

        Return 0 if parsing fails.
        """
        try:
            matched = re.findall(r"(\d+)", evaluator_output)
            return int(matched[-1])
        except (IndexError, ValueError):
            return 0

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if task_inputs_list is None:
            task_inputs_list = [{} for _ in lm_outputs]

        evaluator_input_list: list[str] = []
        for lm_output, task_input, references in zip(
            lm_outputs,
            task_inputs_list,
            references_list,
        ):
            prompt_inputs = {
                "lm_output": lm_output,
                "references": references,
                **task_input,
            }
            evaluator_input = self._prompt_template.embed_input(prompt_inputs)
            evaluator_input_list.append(evaluator_input)

        evaluator_output_list: list[str] = []
        for batch_evaluator_input in batch_iter(
            evaluator_input_list,
            batch_size=self._batch_size,
        ):
            evaluator_outputs = self._language_model.batch_complete_text(
                batch_evaluator_input,
            )
            evaluator_output_list += evaluator_outputs

        evaluator_score_list: list[int] = []
        for evaluator_output in evaluator_output_list:
            evaluator_score = self._parse_evaluator_output(evaluator_output)
            evaluator_score_list.append(evaluator_score)

        average_evaluator_score = sum(evaluator_score_list) / len(evaluator_score_list)
        return MetricResult(
            {"llm_score": average_evaluator_score},
            instance_details=[
                {"llm_score": eval_score, "llm_score_output": eval_out}
                for eval_score, eval_out in zip(
                    evaluator_score_list,
                    evaluator_output_list,
                )
            ],
        )


class ChatLLMScore(Metric):
    """
    A metric that evaluates the output of `LanguageModel.batch_generate_chat_response`.

    Args:
        language_model: An instance of `LanguageModel` to evaluate the output of the model.
        prompt_template: An instance of `PromptTemplate` to embed the input for the evaluator.
        system_message: A system message to be prepended to the input for the evaluator.
        batch_size: The batch size for the evaluator.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        system_message: Optional[Union[str, PromptTemplate]] = None,
        batch_size: int = 4,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._system_message = system_message
        self._batch_size = batch_size

    @staticmethod
    def _parse_evaluator_output(evaluator_output: str) -> int:
        """Extract the last integer value from the evaluator output.

        Return 0 if parsing fails.
        """
        try:
            matched = re.findall(r"(\d+)", evaluator_output)
            return int(matched[-1])
        except (IndexError, ValueError):
            return 0

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if task_inputs_list is None:
            task_inputs_list = [{} for _ in lm_outputs]
        evaluator_input_list: list[list[dict[str, str]]] = []
        for lm_output, task_input, references in zip(
            lm_outputs,
            task_inputs_list,
            references_list,
        ):
            prompt_inputs = {
                "lm_output": lm_output,
                "references": references,
                **task_input,
            }
            evaluator_input = self._prompt_template.embed_input(prompt_inputs)
            input_chat_messages = [{"role": "user", "content": evaluator_input}]
            if self._system_message:
                if isinstance(self._system_message, str):
                    system_message = self._system_message
                else:
                    system_message = self._system_message.embed_input(prompt_inputs)
                input_chat_messages.insert(
                    0,
                    {"role": "system", "content": system_message},
                )
            evaluator_input_list.append(input_chat_messages)

        evaluator_output_list: list[str] = []
        for batch_inputs in batch_iter(
            evaluator_input_list,
            batch_size=self._batch_size,
        ):
            evaluator_outputs = self._language_model.batch_generate_chat_response(
                batch_inputs,
            )
            evaluator_output_list += evaluator_outputs

        evaluator_score_list: list[int] = []
        for evaluator_output in evaluator_output_list:
            evaluator_score = self._parse_evaluator_output(evaluator_output)
            evaluator_score_list.append(evaluator_score)

        average_evaluator_score = sum(evaluator_score_list) / len(evaluator_score_list)
        return MetricResult(
            {"llm_score": average_evaluator_score},
            instance_details=[
                {"llm_score": eval_score, "llm_score_output": eval_out}
                for eval_score, eval_out in zip(
                    evaluator_score_list,
                    evaluator_output_list,
                )
            ],
        )
