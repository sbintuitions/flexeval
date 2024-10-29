from __future__ import annotations

import re
from collections import Counter, defaultdict

from loguru import logger

from flexeval.core.language_model import LanguageModel
from flexeval.core.metric.llm_score import (
    generate_evaluations,
    prepare_chat_input_for_evaluator,
    prepare_text_input_for_evaluator,
)
from flexeval.core.prompt_template import PromptTemplate

from .base import Metric, MetricResult


def parse_label_from_evaluator_output(evaluator_output: str, label_names: list[str]) -> str | None:
    """Extract the last label from the evaluator output.

    Return None if parsing fails.
    """
    pattern = rf"({'|'.join(label_names)})"
    matched = re.findall(pattern, evaluator_output)
    if not matched:
        return None

    parsed_label = matched[-1]
    if parsed_label not in set(label_names):
        return None
    return parsed_label


def calc_label_dist(valid_labels: list[int], label_names: list[str]) -> dict[str, float]:
    counter = Counter(valid_labels)
    dist_dict: dict[str, float] = {}
    for label in label_names:
        dist_dict[label] = counter.get(label, 0.0) / len(valid_labels)
    return dist_dict


def summarize_evaluator_labels(
    evaluator_label_list: list[int | None],
    task_inputs_list: list[dict[str, str]],
    label_names: list[str],
    weights: list[float],
    category_key: str | None = None,
) -> dict[str, float]:
    """Summarize evaluator_score_list. If category_key is given, return
    category-wise mean score as well as overall mean score.
    """
    score_key = "llm_score"
    dist_key = "llm_label_distribution"

    label2point = dict(zip(label_names, weights))
    # compute overall mean score and label distribution
    all_valid_labels: list[int] = [label for label in evaluator_label_list if label is not None]
    score = sum([label2point[label] for label in all_valid_labels])
    score /= len(all_valid_labels)
    dist = calc_label_dist(all_valid_labels, label_names)
    num_failed_score_parses = len(evaluator_label_list) - len(all_valid_labels)
    summary = {score_key: score, dist_key: dist, "num_failed_score_parses": num_failed_score_parses}

    # compute category-wise stats if category_key is given
    category2valid_labels: dict[str, list[str]] = defaultdict(list)
    for label, task_inputs in zip(evaluator_label_list, task_inputs_list):
        if label is None or category_key is None:
            continue
        if category_key in task_inputs:
            category2valid_labels[task_inputs[category_key]].append(label)

    category2mean_score: dict[str, float] = {}
    category2dist: dict[str, float] = {}
    for category, valid_labels in category2valid_labels.items():
        score = sum([label2point[label] for label in valid_labels])
        score /= len(valid_labels)
        category2mean_score[category] = score
        category2dist[category] = calc_label_dist(valid_labels, label_names)

    for category in category2mean_score:
        summary[f"{score_key}/{category}"] = category2mean_score[category]
        summary[f"{dist_key}/{category}"] = category2dist[category]

    return summary


class LLMLabel(Metric):
    """Let LanguageModel to evaluate the output of another LanguageModel.

    You can specify the evaluation criteria in `PromptTemplate`.
    The last label value found in the output of the evaluator is used to compute the evaluation score.
    You can assign a score to each label.
    The final output is the average score and the distribution of the labels.

    Args:
        language_model: An instance of `LanguageModel` to evaluate the output of the model.
        prompt_template: An instance of `PromptTemplate` to embed the input for the evaluator.
        label_names: A list of valid label names.
        label_points: A list of points for each label specified in label_names.
        batch_size: The batch size for the evaluator.
        disable_tqdm: Whether to disable the progress bar.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in task inputs.

    Examples:
        >>> from flexeval import OpenAIChatAPI, Jinja2PromptTemplate, LLMLabel
        >>> language_model = OpenAIChatAPI(model="gpt-3.5-turbo")
        >>> template = "Evaluate the quality of this text on a scale of Good/Bad.\\n`{{ lm_output }}`\\nPut the label at the end like [[Good]]."
        >>> prompt_template = Jinja2PromptTemplate(template)
        >>> label_names = ["Good", "Bad"]
        >>> label_points = [1.0, 0.0]
        >>> llm_label = LLMLabel(language_model, prompt_template, label_names, label_points)
        >>> lm_outputs = ["Hello, world!", "Good mrrrning!"]
        >>> result = llm_label.evaluate(lm_outputs)
        >>> print(result)
        MetricResult(
            summary={'llm_score': 0.5, 'llm_label_distribution': {'Good': 0.5, 'Bad': 0.5}, 'num_failed_score_parses': 0},
            instance_details=[
                {
                    'llm_label': 'Good',
                    'llm_score': 1.0,
                    'llm_label_input': 'Evaluate the quality of this text...',
                    'llm_label_output': 'This text is natural, ... [[Good]]'
                },
                {
                    'llm_label': 'Bad',
                    'llm_score': 0.0,
                    'llm_label_input': 'Evaluate the quality of this text on a scale of Good/Bad.\\n`Good mrrrning!`\\nPut the label at the end like [[Good]].',
                    'llm_label_output': 'This text contains a spelling error, ... [[Bad]]'
                }
            ]
        )
    """  # noqa: E501

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        label_names: list[str],
        label_points: list[float | int] | None = None,
        batch_size: int = 4,
        disable_tqdm: bool = False,
        valid_score_range: tuple[int, int] | None = None,
        category_key: str | None = None,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.label_names = [re.escape(label) for label in label_names]

        if label_points:
            if len(self.label_names) != len(label_points):
                msg = "The lengths of label_names and weights do not match."
                raise ValueError(msg)
            label_points: list[float] = list(map(float, label_points))
        else:
            label_points = [0.0] * len(label_names)
            label_points[0] = 1.0

        self.weights = label_points
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

        evaluator_label_list: list[int | None] = []
        for evaluator_output in evaluator_output_list:
            evaluator_label = parse_label_from_evaluator_output(
                evaluator_output,
                label_names=self.label_names,
            )
            if evaluator_label is None:
                logger.warning(f"Failed to parse label from evaluator output: {evaluator_output}")
            evaluator_label_list.append(evaluator_label)

        label2point = dict(zip(self.label_names, self.weights))
        evaluator_score_list: list[float | None] = [label2point.get(label) for label in evaluator_label_list]

        summary = summarize_evaluator_labels(
            evaluator_label_list,
            task_inputs_list,
            self.label_names,
            self.weights,
            self.category_key,
        )

        return MetricResult(
            summary,
            instance_details=[
                {
                    "llm_label": eval_label,
                    "llm_score": eval_score,
                    "llm_label_input": eval_in,
                    "llm_label_output": eval_out,
                }
                for eval_label, eval_score, eval_in, eval_out in zip(
                    evaluator_label_list,
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


class ChatLLMLabel(Metric):
    """
    A metric that evaluates the output of `LanguageModel.batch_generate_chat_response`.

    Args:
        language_model: An instance of `LanguageModel` to evaluate the output of the model.
        prompt_template: An instance of `PromptTemplate` to embed the input for the evaluator.
        label_names: A list of valid label names.
        label_points: A list of points for each label specified in label_names.
        system_message: A system message to be prepended to the input for the evaluator.
        batch_size: The batch size for the evaluator.
        disable_tqdm: Whether to disable the progress bar.
        category_key: A key to create category-wise mean score.
            The category key is expected to be in task inputs.

    Examples:
        >>> from flexeval import ChatLLMScore, OpenAIChatAPI, Jinja2PromptTemplate
        >>> language_model = OpenAIChatAPI(model_name="gpt-3.5-turbo")
        >>> template = "Evaluate the quality of this text on a scale of Good/Bad.\\n`{{ lm_output }}`\\nPut the label at the end like [[Good]]."
        >>> prompt_template = Jinja2PromptTemplate(template)
        >>> system_message = "This is the system message."
        >>> label_names = ["Good", "Bad"]
        >>> label_points = [1.0, 0.0]
        >>> llm_label = ChatLLMLabel(language_model, prompt_template, label_names, label_points)
        >>> lm_outputs = ["Hello, world!", "Good morning!"]
        >>> result = llm_label.evaluate(lm_outputs)
        >>> print(result)
        MetricResult(
            summary={'llm_score': 0.5, 'llm_label_distribution': {'Good': 0.5, 'Bad': 0.5}, 'num_failed_score_parses': 0},
            instance_details=[
                {
                    'llm_label': 'Good',
                    'llm_score': 1.0,
                    'llm_label_input': 'Evaluate the quality of this text...',
                    'llm_label_output': 'This text is natural, ... [[Good]]'
                },
                {
                    'llm_label': 'Bad',
                    'llm_score': 0.0,
                    'llm_label_input': 'Evaluate the quality of this text on a scale of Good/Bad.\\n`Good mrrrning!`\\nPut the label at the end like [[Good]].',
                    'llm_label_output': 'This text contains a spelling error, ... [[Bad]]'
                }
            ]
        )
    """  # noqa: E501

    def __init__(
        self,
        language_model: LanguageModel,
        prompt_template: PromptTemplate,
        label_names: list[str],
        label_points: list[float | int] | None = None,
        system_message: str | PromptTemplate | None = None,
        batch_size: int = 4,
        disable_tqdm: bool = False,
        category_key: str | None = None,
    ) -> None:
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.label_names = [re.escape(label) for label in label_names]

        if label_points:
            if len(self.label_names) != len(label_points):
                msg = "The lengths of label_names and weights do not match."
                raise ValueError(msg)
            label_points: list[float] = list(map(float, label_points))
        else:
            label_points = [0.0] * len(label_names)
            label_points[0] = 1.0

        self.weights = label_points
        self.system_message = system_message
        self.batch_size = batch_size
        self.disable_tqdm = disable_tqdm
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

        evaluator_label_list: list[str] = []
        for evaluator_output in evaluator_output_list:
            evaluator_label = parse_label_from_evaluator_output(
                evaluator_output,
                label_names=self.label_names,
            )
            if evaluator_label is None:
                logger.warning(f"Failed to parse label from evaluator output: {evaluator_output}")
            evaluator_label_list.append(evaluator_label)

        label2point = dict(zip(self.label_names, self.weights))
        evaluator_score_list: list[float | None] = [label2point.get(label) for label in evaluator_label_list]

        summary = summarize_evaluator_labels(
            evaluator_label_list,
            task_inputs_list,
            self.label_names,
            self.weights,
            self.category_key,
        )

        return MetricResult(
            summary,
            instance_details=[
                {
                    "llm_label": eval_label,
                    "llm_score": eval_score,
                    "llm_label_input": eval_in,
                    "llm_label_output": eval_out,
                }
                for eval_label, eval_score, eval_in, eval_out in zip(
                    evaluator_label_list,
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
