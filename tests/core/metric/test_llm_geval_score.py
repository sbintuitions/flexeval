from __future__ import annotations

import re

import pytest

from flexeval import Jinja2PromptTemplate, LanguageModel
from flexeval.core.metric.llm_geval_score import ChatLLMGEvalScore, LLMGEvalScore, calculate_weighted_average


class EchoBackLanguageModel(LanguageModel):
    def compute_log_probs(
        self,
        text_list: list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> list[float | None]:
        log_prob_list = []
        for text, prefix in zip(text_list, prefix_list):
            if re.match(r"\d+", text):
                index = int(text)
                if (
                    (index == 2 and prefix.startswith("[B]"))
                    or (index == 3 and prefix.startswith("[C]"))
                    or (index == 4 and prefix.startswith("[A]"))
                ):
                    log_prob_list.append(-1)
                elif prefix.startswith(("[A]", "[B]", "[C]")):
                    log_prob_list.append(-2)
                else:
                    log_prob_list.append(None)
            else:
                log_prob_list.append(None)
        return log_prob_list

    def compute_chat_log_probs(
        self, prompt_list: list[list[dict[str, str]]], response_list: list[dict[str, str]]
    ) -> list[float | None]:
        return self.compute_log_probs(
            [response["content"] for response in response_list], [prompt[-1]["content"] for prompt in prompt_list]
        )


@pytest.mark.parametrize(
    ("evaluator_logprobs", "valid_score_range", "prob_threshold", "expected_score"),
    [
        ({"5": 0}, None, 0, 5),
        ({"5": 0}, (0, 3), 0, None),
        ({}, None, 0, None),
        ({"0": -0.5, "1": -1.2, "2": -2.8, "3": -6.6}, (0, 3), 0, 0.44014590056102276),
        ({"0": -0.5, "1": -1.2, "2": -2.8, "3": -6.6}, (0, 3), 0.99, None),  # sum of probs = 0.9698...
    ],
)
def test_calculate_weighted_average(
    evaluator_logprobs: dict[str, float],
    valid_score_range: tuple[int, int] | None,
    prob_threshold: float,
    expected_score: float | None,
) -> None:
    score, _ = calculate_weighted_average(evaluator_logprobs, valid_score_range, prob_threshold)
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

    assert len(metric_output.summary) == 2
    assert metric_output.summary["llm_geval_score"] == pytest.approx(3.0, rel=1e-5)
    assert metric_output.summary["num_failed_score_parses"] == 1

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
    extra_info_list = [
        {"category": "category-0"},
        {"category": "category-1"},
        {"category": "category-0"},
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )

    assert len(metric_output.summary) == 4
    assert metric_output.summary["llm_geval_score"] == pytest.approx(3.0, rel=1e-5)
    assert metric_output.summary["num_failed_score_parses"] == 0
    assert metric_output.summary["llm_geval_score/category-0"] == pytest.approx(3.1278810469948057, rel=1e-5)
    assert metric_output.summary["llm_geval_score/category-1"] == pytest.approx(2.744237906010388, rel=1e-5)

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

    assert len(metric_output.summary) == 2
    assert metric_output.summary["llm_geval_score"] == pytest.approx(3.0, rel=1e-5)
    assert metric_output.summary["num_failed_score_parses"] == 1

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
    extra_info_list = [
        {"category": "category-0"},
        {"category": "category-1"},
        {"category": "category-0"},
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )

    assert len(metric_output.summary) == 4
    assert metric_output.summary["llm_geval_score"] == pytest.approx(3.0, rel=1e-5)
    assert metric_output.summary["num_failed_score_parses"] == 0
    assert metric_output.summary["llm_geval_score/category-0"] == pytest.approx(3.1278810469948057, rel=1e-5)
    assert metric_output.summary["llm_geval_score/category-1"] == pytest.approx(2.744237906010388, rel=1e-5)

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_geval_score_input"] == [{"role": "user", "content": lm_output}]
