from __future__ import annotations

import pytest

from flexeval import Jinja2PromptTemplate, LanguageModel
from flexeval.core.language_model.base import LMOutput
from flexeval.core.metric.llm_label import ChatLLMLabel, LLMLabel, parse_label_from_evaluator_output


class EchoBackLanguageModel(LanguageModel):
    def complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        return [LMOutput(text=text, finish_reason="length") for text in text_list]

    def generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[LMOutput]:
        return [LMOutput(text=mes[-1]["content"], finish_reason="length") for mes in chat_messages_list]


@pytest.mark.parametrize(
    ("evaluator_output", "label_names", "expected_label"),
    [
        ("The label is A.", ["A", "B", "C"], "A"),
        ("The label is Z.", ["A", "B", "C"], None),
    ],
)
def test_parse_label_from_evaluator_output(
    evaluator_output: str,
    label_names: tuple[int, int] | None,
    expected_label: str | None,
) -> None:
    label = parse_label_from_evaluator_output(evaluator_output, label_names)
    assert label == expected_label


@pytest.mark.parametrize("metric_prefix", [None, "prefix"])
def test_llm_label(metric_prefix: str | None) -> None:
    metric = LLMLabel(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        label_names=["Good", "Neutral", "Bad"],
        label_points=[1.0, 0.5, 0.0],
        metric_prefix=metric_prefix,
    )
    lm_outputs = ["This is Good.", "This is Neutral.", "This is Bad.", "This is Great."]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
    )

    metric_prefix = metric_prefix + "-" if metric_prefix else ""
    assert metric_output.summary == {
        f"{metric_prefix}llm_score": 0.5,
        f"{metric_prefix}num_failed_score_parses": 1,
        f"{metric_prefix}llm_label_distribution": {"Good": 1 / 3, "Neutral": 1 / 3, "Bad": 1 / 3},
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail[f"{metric_prefix}llm_label_input"] == lm_output
        assert instance_detail[f"{metric_prefix}llm_label_output"] == lm_output


def test_llm_label_with_category() -> None:
    metric = LLMLabel(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        label_names=["Good", "Neutral", "Bad"],
        label_points=[1.0, 0.5, 0.0],
        category_key="category",
    )
    lm_outputs = ["This is Good.", "This is Neutral.", "This is Bad.", "This is Great."]
    extra_info_list = [
        {"category": "category-0"},
        {"category": "category-0"},
        {"category": "category-1"},
        {"category": "category-2"},
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )

    assert metric_output.summary == {
        "llm_score": 0.5,
        "llm_label_distribution": {"Good": 1 / 3, "Neutral": 1 / 3, "Bad": 1 / 3},
        "num_failed_score_parses": 1,
        "llm_score/category-0": 0.75,
        "llm_label_distribution/category-0": {"Good": 1 / 2, "Neutral": 1 / 2, "Bad": 0 / 2},
        "llm_score/category-1": 0.0,
        "llm_label_distribution/category-1": {"Good": 0 / 1, "Neutral": 0 / 1, "Bad": 1 / 1},
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_label_input"] == lm_output
        assert instance_detail["llm_label_output"] == lm_output


@pytest.mark.parametrize("metric_prefix", [None, "prefix"])
def test_chat_llm_label(metric_prefix: str | None) -> None:
    metric = ChatLLMLabel(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        label_names=["Good", "Neutral", "Bad"],
        label_points=[1.0, 0.5, 0.0],
        metric_prefix=metric_prefix,
    )
    lm_outputs = ["This is Good.", "This is Neutral.", "This is Bad.", "This is Great."]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
    )

    metric_prefix = metric_prefix + "-" if metric_prefix else ""
    assert metric_output.summary == {
        f"{metric_prefix}llm_score": 0.5,
        f"{metric_prefix}num_failed_score_parses": 1,
        f"{metric_prefix}llm_label_distribution": {"Good": 1 / 3, "Neutral": 1 / 3, "Bad": 1 / 3},
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail[f"{metric_prefix}llm_label_input"] == [{"role": "user", "content": lm_output}]
        assert instance_detail[f"{metric_prefix}llm_label_output"] == lm_output


def test_chat_llm_label_with_category() -> None:
    metric = ChatLLMLabel(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        label_names=["Good", "Neutral", "Bad"],
        label_points=[1.0, 0.5, 0.0],
        category_key="category",
    )
    lm_outputs = ["This is Good.", "This is Neutral.", "This is Bad.", "This is Great."]
    extra_info_list = [
        {"category": "category-0"},
        {"category": "category-0"},
        {"category": "category-1"},
        {"category": "category-2"},
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )

    assert metric_output.summary == {
        "llm_score": 0.5,
        "llm_label_distribution": {"Good": 1 / 3, "Neutral": 1 / 3, "Bad": 1 / 3},
        "num_failed_score_parses": 1,
        "llm_score/category-0": 0.75,
        "llm_label_distribution/category-0": {"Good": 1 / 2, "Neutral": 1 / 2, "Bad": 0 / 2},
        "llm_score/category-1": 0.0,
        "llm_label_distribution/category-1": {"Good": 0 / 1, "Neutral": 0 / 1, "Bad": 1 / 1},
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_label_input"] == [{"role": "user", "content": lm_output}]
        assert instance_detail["llm_label_output"] == lm_output
