from __future__ import annotations

import re

import tqdm

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

    Examples:
        >>> from flexeval import LLMScore, OpenAIChatAPI, Jinja2PromptTemplate
        >>> language_model = OpenAIChatAPI(model_name="gpt-3.5-turbo")
        >>> template = "Evaluate the quality of this text.\\n`{{ lm_output }}`\\nPut the score at the end like [[5]]."
        >>> prompt_template = Jinja2PromptTemplate(template)
        >>> llm_score = LLMScore(language_model, prompt_template)
        >>> lm_outputs = ["Hello, world!", "Good morning!"]
        >>> result = llm_score.evaluate(lm_outputs)
        >>> print(result)
        MetricResult(
            summary={'llm_score': 3.0},
            instance_details=[
                {
                    'llm_score': 2,
                    'llm_score_input': 'Evaluate the quality of this text...',
                    'llm_score_output': 'This text is very simple,... Therefore, its quality is average. [[2]]'},
                {
                    'llm_score': 4,
                    'llm_score_input': 'Evaluate the quality of this text...',
                    'llm_score_output': '... Overall, the quality of the text is good but basic. [[4]]'}
            ]
        )
    """

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        batch_size: int = 4,
        disable_tqdm: bool = False,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.disable_tqdm = disable_tqdm

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
        references_list: list[list[str]] | None = None,
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if task_inputs_list is None:
            task_inputs_list = [{} for _ in lm_outputs]
        if references_list is None:
            references_list = [[] for _ in lm_outputs]

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
            evaluator_input = self.prompt_template.embed_inputs(prompt_inputs)
            evaluator_input_list.append(evaluator_input)

        with tqdm.tqdm(
            total=len(evaluator_input_list),
            disable=self.disable_tqdm,
            desc="Calculating LLM score",
        ) as pbar:
            evaluator_output_list: list[str] = []
            for batch_evaluator_input in batch_iter(
                evaluator_input_list,
                batch_size=self.batch_size,
            ):
                evaluator_outputs = self.language_model.batch_complete_text(
                    batch_evaluator_input,
                )
                evaluator_output_list += evaluator_outputs
                pbar.update(len(batch_evaluator_input))

        evaluator_score_list: list[int] = []
        for evaluator_output in evaluator_output_list:
            evaluator_score = self._parse_evaluator_output(evaluator_output)
            evaluator_score_list.append(evaluator_score)

        average_evaluator_score = sum(evaluator_score_list) / len(evaluator_score_list)
        return MetricResult(
            {"llm_score": average_evaluator_score},
            instance_details=[
                {"llm_score": eval_score, "llm_score_input": eval_in, "llm_score_output": eval_out}
                for eval_score, eval_in, eval_out in zip(
                    evaluator_score_list,
                    evaluator_input_list,
                    evaluator_output_list,
                )
            ],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(language_model={self.language_model}, prompt_template={self.prompt_template})"
        )


class ChatLLMScore(Metric):
    """
    A metric that evaluates the output of `LanguageModel.batch_generate_chat_response`.

    Args:
        language_model: An instance of `LanguageModel` to evaluate the output of the model.
        prompt_template: An instance of `PromptTemplate` to embed the input for the evaluator.
        system_message: A system message to be prepended to the input for the evaluator.
        batch_size: The batch size for the evaluator.

    Examples:
        >>> from flexeval import ChatLLMScore, OpenAIChatAPI, Jinja2PromptTemplate
        >>> language_model = OpenAIChatAPI(model_name="gpt-3.5-turbo")
        >>> template = "Evaluate the quality of this text.\\n`{{ lm_output }}`\\nPut the score at the end like [[5]]."
        >>> prompt_template = Jinja2PromptTemplate(template)
        >>> system_message = "This is the system message."
        >>> llm_score = ChatLLMScore(language_model, prompt_template, system_message)
        >>> lm_outputs = ["Hello, world!", "Good morning!"]
        >>> result = llm_score.evaluate(lm_outputs)
        >>> print(result)
        MetricResult(
            summary={'llm_score': 3.0},
            instance_details=[
                {
                    'llm_score': 2,
                    'llm_score_input': [{'role': 'user', 'content': 'Evaluate the quality of this text...'}],
                    'llm_score_output': 'This text is very simple,... Therefore, its quality is average. [[2]]'},
                {
                    'llm_score': 4,
                    'llm_score_input': [{'role': 'user', 'content': 'Evaluate the quality of this text...'}],
                    'llm_score_output': '... Overall, the quality of the text is good but basic. [[4]]'}
            ]
        )
    """

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        system_message: str | PromptTemplate | None = None,
        batch_size: int = 4,
        disable_tqdm: bool = False,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.system_message = system_message
        self.batch_size = batch_size
        self.disable_tqdm = disable_tqdm

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
        references_list: list[list[str]] | None = None,
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if task_inputs_list is None:
            task_inputs_list = [{} for _ in lm_outputs]
        if references_list is None:
            references_list = [[] for _ in lm_outputs]
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
            evaluator_input = self.prompt_template.embed_inputs(prompt_inputs)
            input_chat_messages = [{"role": "user", "content": evaluator_input}]
            if self.system_message:
                if isinstance(self.system_message, str):
                    system_message = self.system_message
                else:
                    system_message = self.system_message.embed_inputs(prompt_inputs)
                input_chat_messages.insert(
                    0,
                    {"role": "system", "content": system_message},
                )
            evaluator_input_list.append(input_chat_messages)

        with tqdm.tqdm(
            total=len(evaluator_input_list),
            disable=self.disable_tqdm,
            desc="Calculating ChatLLM score",
        ) as pbar:
            evaluator_output_list: list[str] = []
            for batch_inputs in batch_iter(
                evaluator_input_list,
                batch_size=self.batch_size,
            ):
                evaluator_outputs = self.language_model.batch_generate_chat_response(
                    batch_inputs,
                )
                evaluator_output_list += evaluator_outputs
                pbar.update(len(batch_inputs))

        evaluator_score_list: list[int] = []
        for evaluator_output in evaluator_output_list:
            evaluator_score = self._parse_evaluator_output(evaluator_output)
            evaluator_score_list.append(evaluator_score)

        average_evaluator_score = sum(evaluator_score_list) / len(evaluator_score_list)
        return MetricResult(
            {"llm_score": average_evaluator_score},
            instance_details=[
                {"llm_score": eval_score, "llm_score_input": eval_in, "llm_score_output": eval_out}
                for eval_score, eval_in, eval_out in zip(
                    evaluator_score_list,
                    evaluator_input_list,
                    evaluator_output_list,
                )
            ],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(language_model={self.language_model}, prompt_template={self.prompt_template})"
        )
