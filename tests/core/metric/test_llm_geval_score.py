from __future__ import annotations

import re

import pytest

from flexeval import Jinja2PromptTemplate, LanguageModel
from flexeval.core.language_model.base import LMOutput
from flexeval.core.metric.llm_geval_score import ChatLLMGEvalScore, LLMGEvalScore, calculate_weighted_average
from flexeval.core.metric.utils import extract_text_from_outputs


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


@pytest.mark.parametrize(
    ("lm_outputs", "extra_info_list", "expected_summary"),
    [
        (
            [
                "[A] Output a number from 1 to 5.",
                "[B] Output a number from 1 to 5.",
                "[C] Output a number from 1 to 5.",
                "[D] Output a number from 1 to 5.",
            ],
            None,
            {
                "llm_geval_score": pytest.approx(3.0, rel=1e-5),
                "num_failed_score_parses": 1,
            },
        ),
        (
            [
                "[A] Output a number from 1 to 5.",
                "[B] Output a number from 1 to 5.",
                "[C] Output a number from 1 to 5.",
            ],
            [
                {"category": "category-0"},
                {"category": "category-1"},
                {"category": "category-0"},
            ],
            {
                "llm_geval_score": pytest.approx(3.0, rel=1e-5),
                "num_failed_score_parses": 0,
                "llm_geval_score/category-0": pytest.approx(3.1278810469948057, rel=1e-5),
                "llm_geval_score/category-1": pytest.approx(2.744237906010388, rel=1e-5),
            },
        ),
    ],
    indirect=["lm_outputs"],
)
@pytest.mark.parametrize("metric_prefix", ["", "prefix"])
def test_llm_geval_score(
    lm_outputs: list[str | LMOutput],
    extra_info_list: list[dict[str, str]] | None,
    expected_summary: dict[str, float | int],
    metric_prefix: str,
) -> None:
    metric = LLMGEvalScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        valid_score_range=(1, 5),
        category_key="category" if extra_info_list else None,
        metric_prefix=metric_prefix,
    )
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )

    if metric_prefix:
        metric_prefix += "-"

    expected_len = len(expected_summary)
    assert len(metric_output.summary) == expected_len

    for key, value in expected_summary.items():
        assert metric_output.summary[f"{metric_prefix}{key}"] == value

    for lm_output, instance_detail in zip(extract_text_from_outputs(lm_outputs), metric_output.instance_details):
        assert instance_detail[f"{metric_prefix}llm_geval_score_input"] == lm_output


@pytest.mark.parametrize(
    ("lm_outputs", "extra_info_list", "expected_summary"),
    [
        (
            [
                "[A] Output a number from 1 to 5.",
                "[B] Output a number from 1 to 5.",
                "[C] Output a number from 1 to 5.",
                "[D] Output a number from 1 to 5.",
            ],
            None,
            {
                "llm_geval_score": pytest.approx(3.0, rel=1e-5),
                "num_failed_score_parses": 1,
            },
        ),
        (
            [
                "[A] Output a number from 1 to 5.",
                "[B] Output a number from 1 to 5.",
                "[C] Output a number from 1 to 5.",
            ],
            [
                {"category": "category-0"},
                {"category": "category-1"},
                {"category": "category-0"},
            ],
            {
                "llm_geval_score": pytest.approx(3.0, rel=1e-5),
                "num_failed_score_parses": 0,
                "llm_geval_score/category-0": pytest.approx(3.1278810469948057, rel=1e-5),
                "llm_geval_score/category-1": pytest.approx(2.744237906010388, rel=1e-5),
            },
        ),
    ],
    indirect=["lm_outputs"],
)
@pytest.mark.parametrize("metric_prefix", ["", "prefix"])
def test_chat_llm_geval_score(
    lm_outputs: list[str | LMOutput],
    extra_info_list: list[dict[str, str]] | None,
    expected_summary: dict[str, float | int],
    metric_prefix: str,
) -> None:
    metric = ChatLLMGEvalScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        valid_score_range=(1, 5),
        category_key="category" if extra_info_list else None,
        metric_prefix=metric_prefix,
    )
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )

    if metric_prefix:
        metric_prefix += "-"

    expected_len = len(expected_summary)
    assert len(metric_output.summary) == expected_len

    for key, value in expected_summary.items():
        assert metric_output.summary[f"{metric_prefix}{key}"] == value

    for lm_output, instance_detail in zip(extract_text_from_outputs(lm_outputs), metric_output.instance_details):
        assert instance_detail[f"{metric_prefix}llm_geval_score_input"] == [{"role": "user", "content": lm_output}]
