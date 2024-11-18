from __future__ import annotations

import re
from collections import defaultdict

import tqdm
from loguru import logger

from flexeval.core.language_model import LanguageModel
from flexeval.core.prompt_template import PromptTemplate
from flexeval.core.utils.data_util import batch_iter

from .base import Metric, MetricResult


def parse_score_from_evaluator_output(evaluator_output: str, valid_score_range: tuple[int, int] | None) -> int | None:
    """Extract the last integer value from the evaluator output.

    Return None if parsing fails.
    """
    matched = re.findall(r"(\d+)", evaluator_output)
    if not matched:
        return None

    parsed_score = int(matched[-1])
    if valid_score_range and not valid_score_range[0] <= parsed_score <= valid_score_range[1]:
        return None
    return parsed_score


def summarize_evaluator_scores(
    evaluator_score_list: list[int | None],
    task_inputs_list: list[dict[str, str]],
    category_key: str | None = None,
) -> dict[str, float]:
    """Summarize evaluator_score_list. If category_key is given, return
    category-wise mean score as well as overall mean score.
    """

    # compute overall mean score
    all_valid_scores: list[int] = [s for s in evaluator_score_list if s is not None]
    llm_score = sum(all_valid_scores) / len(all_valid_scores)
    num_failed_score_parses = len(evaluator_score_list) - len(all_valid_scores)
    summary = {"llm_score": llm_score, "num_failed_score_parses": num_failed_score_parses}

    # compute category-wise mean score if category_key is given
    category2valid_scores: dict[str, list[int]] = defaultdict(list)
    for score, task_inputs in zip(evaluator_score_list, task_inputs_list):
        if score is None or category_key is None:
            continue
        if category_key in task_inputs:
            category2valid_scores[task_inputs[category_key]].append(score)

    category2mean_score: dict[str, float] = {}
    for category, valid_scores in category2valid_scores.items():
        category2mean_score[category] = sum(valid_scores) / len(valid_scores)

    for category, mean_score in category2mean_score.items():
        summary[f"llm_score/{category}"] = mean_score
    return summary


def prepare_text_input_for_evaluator(
    lm_outputs: list[str],
    references_list: list[list[str]],
    task_inputs_list: list[dict[str, str]],
    prompt_template: PromptTemplate,
) -> list[str]:
    """Create input texts for the evaluator
    by integrating the task inputs, the model outputs, and the prompt template for evaluator.
    """

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
        evaluator_input = prompt_template.embed_inputs(prompt_inputs)
        evaluator_input_list.append(evaluator_input)
    return evaluator_input_list


def prepare_chat_input_for_evaluator(
    lm_outputs: list[str],
    references_list: list[list[str]],
    task_inputs_list: list[dict[str, str]],
    prompt_template: PromptTemplate,
    system_message: str | PromptTemplate | None = None,
) -> list[list[dict[str, str]]]:
    """Create input chat messages for the evaluator
    by integrating the task inputs, the model outputs, and the prompt template for evaluator.
    """

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
        evaluator_input = prompt_template.embed_inputs(prompt_inputs)
        input_chat_messages = [{"role": "user", "content": evaluator_input}]
        if system_message:
            if not isinstance(system_message, str):
                system_message_rendered = system_message.embed_inputs(prompt_inputs)
            else:
                system_message_rendered = system_message
            input_chat_messages.insert(
                0,
                {"role": "system", "content": system_message_rendered},
            )
        evaluator_input_list.append(input_chat_messages)
    return evaluator_input_list


def generate_evaluations(
    evaluator_input_list: list[str] | list[list[dict[str, str]]],
    language_model: LanguageModel,
    batch_size: int,
    disable_tqdm: bool = False,
    desc_for_tqdm: str | None = None,
) -> list[str]:
    """Generate evaluation texts for each input in evaluator_input_list.

    - If evaluator_input_list contains a list of plain texts, use
      language_model.batch_complete_text() to generate evaluation outputs.
    - If evaluator_input_list contains a list of chat message dictionaries,
      use language_model.batch_generate_chat_response().
    """

    with tqdm.tqdm(
        total=len(evaluator_input_list),
        disable=disable_tqdm,
        desc=desc_for_tqdm,
    ) as pbar:
        evaluator_output_list: list[str] = []
        for batch_inputs in batch_iter(
            evaluator_input_list,
            batch_size=batch_size,
        ):
            if all(isinstance(elem, str) for elem in batch_inputs):
                evaluator_outputs = language_model.batch_complete_text(
                    batch_inputs,
                )
            else:
                evaluator_outputs = language_model.batch_generate_chat_response(
                    batch_inputs,
                )
            evaluator_output_list += evaluator_outputs
            pbar.update(len(batch_inputs))
    return evaluator_output_list


class LLMScore(Metric):
    """Let LanguageModel to evaluate the output of another LanguageModel.

    You can specify the evaluation criteria in `PromptTemplate`.
    The last integer value in the output of the evaluator is used as the evaluation score.

    Args:
        language_model: An instance of `LanguageModel` to evaluate the output of the model.
        prompt_template: An instance of `PromptTemplate` to embed the input for the evaluator.
        batch_size: The batch size for the evaluator.
        disable_tqdm: Whether to disable the progress bar.
        valid_score_range: A tuple of two integers representing the valid score range.
            If the parsed score is out of the range, it will be ignored.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in task inputs.

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
            summary={'llm_score': 3.0, 'num_failed_score_parses': 0},
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
        valid_score_range: tuple[int, int] | None = None,
        category_key: str | None = None,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.disable_tqdm = disable_tqdm
        self.valid_score_range = valid_score_range
        self.category_key = category_key

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

        evaluator_input_list: list[str] = prepare_text_input_for_evaluator(
            lm_outputs, references_list, task_inputs_list, self.prompt_template
        )
        evaluator_output_list: list[str] = generate_evaluations(
            evaluator_input_list, self.language_model, self.batch_size, self.disable_tqdm, "Calculating LLM score"
        )

        evaluator_score_list: list[int | None] = []
        for evaluator_output in evaluator_output_list:
            evaluator_score = parse_score_from_evaluator_output(
                evaluator_output,
                valid_score_range=self.valid_score_range,
            )
            if evaluator_score is None:
                logger.warning(f"Failed to parse score from evaluator output: {evaluator_output}")
            evaluator_score_list.append(evaluator_score)

        summary = summarize_evaluator_scores(
            evaluator_score_list,
            task_inputs_list,
            self.category_key,
        )

        return MetricResult(
            summary,
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
        disable_tqdm: Whether to disable the progress bar.
        valid_score_range: A tuple of two integers representing the valid score range.
            If the parsed score is out of the range, it will be ignored.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in task inputs.

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
            summary={'llm_score': 3.0, 'num_failed_score_parses': 0},
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
        valid_score_range: tuple[int, int] | None = None,
        category_key: str | None = None,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.system_message = system_message
        self.batch_size = batch_size
        self.disable_tqdm = disable_tqdm
        self.valid_score_range = valid_score_range
        self.category_key = category_key

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

        evaluator_input_list = prepare_chat_input_for_evaluator(
            lm_outputs, references_list, task_inputs_list, self.prompt_template, self.system_message
        )
        evaluator_output_list: list[str] = generate_evaluations(
            evaluator_input_list, self.language_model, self.batch_size, self.disable_tqdm, "Calculating ChatLLM score"
        )

        evaluator_score_list: list[int] = []
        for evaluator_output in evaluator_output_list:
            evaluator_score = parse_score_from_evaluator_output(
                evaluator_output,
                valid_score_range=self.valid_score_range,
            )
            if evaluator_score is None:
                logger.warning(f"Failed to parse score from evaluator output: {evaluator_output}")
            evaluator_score_list.append(evaluator_score)

        summary = summarize_evaluator_scores(
            evaluator_score_list,
            task_inputs_list,
            self.category_key,
        )

        return MetricResult(
            summary,
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
