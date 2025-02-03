from __future__ import annotations

import pytest

from flexeval import Jinja2PromptTemplate, LanguageModel
from flexeval.core.metric.llm_geval_score import ChatLLMGEvalScore, LLMGEvalScore, calculate_weighted_average


class EchoBackLanguageModel(LanguageModel):
    def batch_compute_log_probs(
        self,
        text_list: list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> list[float]:
        if text_list[0].startswith("[A]"):
            return [-2, -2, -2, -1, -2]
        if text_list[0].startswith("[B]"):
            return [-2, -1, -2, -2, -2]
        if text_list[0].startswith("[C]"):
            return [-2, -2, -1, -2, -2]

        # For OpenAI models, we can obtain 20 or less tokens and their logprobs.
        # This simulates all of the valid scores are not obtained from logprob results.
        return [None, None, None, None, None]

    def batch_compute_chat_log_probs(
        self, prompt_list: list[list[dict[str, str]]], response_list: list[dict[str, str]]
    ) -> list[float]:
        text = prompt_list[0][-1]["content"]
        if text.startswith("[A]"):
            return [-2, -2, -2, -1, -2]
        if text.startswith("[B]"):
            return [-2, -1, -2, -2, -2]
        if text.startswith("[C]"):
            return [-2, -2, -1, -2, -2]

        # For OpenAI models, we can obtain 20 or less tokens and their logprobs.
        # This simulates all of the valid scores are not obtained from logprob results.
        return [None, None, None, None, None]


@pytest.mark.parametrize(
    ("evaluator_logprobs", "valid_score_range", "expected_score"),
    [
        ({"5": 0}, None, 0, 5),
        ({"5": 0}, (0, 3), 0, None),
        ({}, None, 0, None),
        ({"0": -0.5, "1": -1.2, "2": -2.8, "3": -6.6}, (0, 3), 0.44014590056102276),
    ],
)
def test_calculate_weighted_average(
    evaluator_logprobs: dict[str, float],
    valid_score_range: tuple[int, int] | None,
    expected_score: float | None,
) -> None:
    score = calculate_weighted_average(evaluator_logprobs, valid_score_range, prob_threshold)
    assert score == expected_score


def test_llm_geval_score() -> None:
    metric = LLMGEvalScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        valid_score_range=(1, 5),
    )
    lm_outputs = [
        "[A] Output a number from 1 to 5.",
        "[B] Output a number from 1 to 5.",
        "[C] Output a number from 1 to 5.",
        "[D] Output a number from 1 to 5.",
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
    )

    assert metric_output.summary == {"llm_geval_score": 3.0, "num_failed_score_parses": 1}

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_geval_score_input"] == lm_output


def test_llm_geval_score_with_category() -> None:
    metric = LLMGEvalScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        valid_score_range=(1, 5),
        category_key="category",
    )
    lm_outputs = [
        "[A] Output a number from 1 to 5.",
        "[B] Output a number from 1 to 5.",
        "[C] Output a number from 1 to 5.",
    ]
    task_inputs_list = [
        {"category": "category-0"},
        {"category": "category-1"},
        {"category": "category-0"},
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        task_inputs_list=task_inputs_list,
    )

    assert metric_output.summary == {
        "llm_geval_score": 3.0,
        "num_failed_score_parses": 0,
        "llm_geval_score/category-0": 3.1278810469948057,
        "llm_geval_score/category-1": 2.744237906010388,
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_geval_score_input"] == lm_output


def test_chat_llm_geval_score() -> None:
    metric = ChatLLMGEvalScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        valid_score_range=(1, 5),
    )
    lm_outputs = [
        "[A] Output a number from 1 to 5.",
        "[B] Output a number from 1 to 5.",
        "[C] Output a number from 1 to 5.",
        "[D] Output a number from 1 to 5.",
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
    )

    assert metric_output.summary == {"llm_geval_score": 3.0, "num_failed_score_parses": 1}

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_geval_score_input"] == [{"role": "user", "content": lm_output}]


def test_chat_llm_geval_score_with_category() -> None:
    metric = ChatLLMGEvalScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        valid_score_range=(1, 5),
        category_key="category",
    )
    lm_outputs = [
        "[A] Output a number from 1 to 5.",
        "[B] Output a number from 1 to 5.",
        "[C] Output a number from 1 to 5.",
    ]
    task_inputs_list = [
        {"category": "category-0"},
        {"category": "category-1"},
        {"category": "category-0"},
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        task_inputs_list=task_inputs_list,
    )

    assert metric_output.summary == {
        "llm_geval_score": 3.0,
        "num_failed_score_parses": 0,
        "llm_geval_score/category-0": 3.1278810469948057,
        "llm_geval_score/category-1": 2.744237906010388,
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_geval_score_input"] == [{"role": "user", "content": lm_output}]
