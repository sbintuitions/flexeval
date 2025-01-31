from __future__ import annotations

import re
from math import exp

import tqdm
from loguru import logger
from numpy import average

from flexeval.core.language_model import LanguageModel
from flexeval.core.prompt_template import PromptTemplate
from flexeval.core.utils.data_util import batch_iter

from .base import Metric, MetricResult
from .llm_score import prepare_chat_input_for_evaluator, prepare_text_input_for_evaluator, summarize_evaluator_scores


def calculate_weighted_average(
    evaluator_logprobs: dict[str, float], valid_score_range: tuple[int, int] | None
) -> float | None:
    """ddd"""
    score_list: list[int] = []
    prob_list: list[float] = []
    for token, logprob in evaluator_logprobs.items():
        matched = re.match(r"(\d)+", token)
        if not matched:
            continue

        parsed_score = int(token)
        if valid_score_range and not valid_score_range[0] <= parsed_score <= valid_score_range[1]:
            continue

        probability = exp(logprob)
        score_list.append(parsed_score)
        prob_list.append(probability)

    if len(score_list) == 0:
        return None

    return average(score_list, weights=prob_list)


def generate_evaluation_logprobs(
    evaluator_input_list: list[str] | list[list[dict[str, str]]],
    language_model: LanguageModel,
    valid_labels: list[str],
    batch_size: int,
    disable_tqdm: bool = False,
    desc_for_tqdm: str | None = None,
) -> list[dict[str, float]]:
    """Generate evaluation logprobs for each input in evaluator_input_list.

    - If evaluator_input_list contains a list of plain texts, use
      language_model.batch_compute_log_probs() to generate evaluation logprobs.
    - If evaluator_input_list contains a list of chat message dictionaries,
      use language_model.batch_compute_chat_log_probs().
    """

    with tqdm.tqdm(
        total=len(evaluator_input_list),
        disable=disable_tqdm,
        desc=desc_for_tqdm,
    ) as pbar:
        evaluator_logprobs_list: list[dict[str, float]] = []
        for batch_inputs in batch_iter(
            evaluator_input_list,
            batch_size=batch_size,
        ):
            if all(isinstance(elem, str) for elem in batch_inputs):
                evaluator_logprobs = language_model.batch_compute_log_probs(
                    [batch_inputs] * len(valid_labels),  # we have to provide n same inputs for generate logprob
                    valid_labels,  # for openai models, len(valid_labels) <= 20 due to constraint
                )
            else:
                evaluator_logprobs = language_model.batch_compute_chat_log_probs(
                    [batch_inputs for _ in valid_labels],
                    [{"role": "assistant", "content": label} for label in valid_labels],
                )
            evaluator_logprobs_list += [dict(zip(valid_labels, evaluator_logprobs))]
            pbar.update(len(batch_inputs))
    return evaluator_logprobs_list


class LLMGEvalScore(Metric):
    """
    docstring
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

        self.valid_labels = [str(score) for score in range(valid_score_range[0], valid_score_range[1] + 1)]

    def evaluate(
        self,
        lm_outputs: list[str],  # 評価対象
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
        evaluator_logprobs_list: list[dict[str, float]] = generate_evaluation_logprobs(
            evaluator_input_list,
            self.language_model,
            self.valid_labels,
            self.batch_size,
            self.disable_tqdm,
            "Calculating logprobs",
        )

        evaluator_score_list: list[int | None] = []
        for evaluator_logprobs in evaluator_logprobs_list:
            evaluator_score = calculate_weighted_average(
                evaluator_logprobs,
                valid_score_range=self.valid_score_range,
            )
            if evaluator_score is None:
                logger.warning(f"Failed to parse score from evaluator logprobs: {evaluator_logprobs}")
            evaluator_score_list.append(evaluator_score)

        summary = summarize_evaluator_scores(
            evaluator_score_list,
            task_inputs_list,
            self.category_key,
        )

        return MetricResult(
            summary,
            instance_details=[
                {
                    "llm_geval_score": eval_score,
                    "llm_geval_score_input": eval_in,
                    "llm_geval_score_logprobs": eval_logprobs,
                }
                for eval_score, eval_in, eval_logprobs in zip(
                    evaluator_score_list,
                    evaluator_input_list,
                    evaluator_logprobs_list,
                )
            ],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(language_model={self.language_model}, prompt_template={self.prompt_template})"
        )


class ChatLLMScore(Metric):
    """ """

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

        self.valid_labels = [str(score) for score in range(valid_score_range[0], valid_score_range[1] + 1)]

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
        evaluator_logprobs_list: list[dict[str, float]] = generate_evaluation_logprobs(
            evaluator_input_list,
            self.language_model,
            self.valid_labels,
            self.batch_size,
            self.disable_tqdm,
            "Calculating logprobs",
        )

        evaluator_score_list: list[int | None] = []
        for evaluator_logprobs in evaluator_logprobs_list:
            evaluator_score = calculate_weighted_average(
                evaluator_logprobs,
                valid_score_range=self.valid_score_range,
            )
            if evaluator_score is None:
                logger.warning(f"Failed to parse score from evaluator logprobs: {evaluator_logprobs}")
            evaluator_score_list.append(evaluator_score)

        summary = summarize_evaluator_scores(
            evaluator_score_list,
            task_inputs_list,
            self.category_key,
        )

        return MetricResult(
            summary,
            instance_details=[
                {
                    "llm_geval_score": eval_score,
                    "llm_geval_score_input": eval_in,
                    "llm_geval_score_logprobs": eval_logprobs,
                }
                for eval_score, eval_in, eval_logprobs in zip(
                    evaluator_score_list,
                    evaluator_input_list,
                    evaluator_logprobs_list,
                )
            ],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(language_model={self.language_model}, prompt_template={self.prompt_template})"
        )
