from __future__ import annotations

import pytest

from flexeval import Jinja2PromptTemplate, LanguageModel
from flexeval.core.metric.llm_score import ChatLLMScore, LLMScore, parse_score_from_evaluator_output


class EchoBackLanguageModel(LanguageModel):
    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        return text_list

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str]:
        return [chat_messages[-1]["content"] for chat_messages in chat_messages_list]


@pytest.mark.parametrize(
    ("evaluator_output", "valid_score_range", "expected_score"),
    [
        ("The final score is 65.", None, 65),
        ("The final score is 65.", (0, 5), None),
        ("Yes, this is a good one.", None, None),
        ("The score is 5. No, it is 6.", None, 6),
    ],
)
def test_parse_score_from_evaluator_output(
    evaluator_output: str,
    valid_score_range: tuple[int, int] | None,
    expected_score: int,
) -> None:
    score = parse_score_from_evaluator_output(evaluator_output, valid_score_range)
    assert score == expected_score


def test_llm_score() -> None:
    metric = LLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
    )
    lm_outputs = ["This score is 1.", "This score is 2.", "This is a good one."]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
    )

    assert metric_output.summary == {"llm_score": 1.5, "num_failed_score_parses": 1}

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_score_input"] == lm_output
        assert instance_detail["llm_score_output"] == lm_output


def test_llm_score_with_category() -> None:
    metric = LLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
    )
    lm_outputs = ["This score is 1.", "This score is 2.", "This is a good one."]
    task_inputs_list = [
        {"category": "category-0"},
        {"category": "category-0"},
        {"category": "category-1"}
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        task_inputs_list=task_inputs_list
    )

    assert metric_output.summary == {
        "llm_score": 1.5,
        "num_failed_score_parses": 1,
        "llm_score/category-0": 1.5
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_score_input"] == lm_output
        assert instance_detail["llm_score_output"] == lm_output


def test_chat_llm_score() -> None:
    metric = ChatLLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
    )
    lm_outputs = ["This score is 1.", "This score is 2.", "This is a good one."]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
    )

    assert metric_output.summary == {"llm_score": 1.5, "num_failed_score_parses": 1}

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_score_input"] == [{"role": "user", "content": lm_output}]
        assert instance_detail["llm_score_output"] == lm_output


def test_chat_llm_score_with_category() -> None:
    metric = ChatLLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
    )
    lm_outputs = ["This score is 1.", "This score is 2.", "This is a good one."]
    task_inputs_list = [
        {"category": "category-0"},
        {"category": "category-0"},
        {"category": "category-1"}
    ]
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        task_inputs_list=task_inputs_list
    )

    assert metric_output.summary == {
        "llm_score": 1.5,
        "num_failed_score_parses": 1,
        "llm_score/category-0": 1.5
    }

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_score_input"] == [{"role": "user", "content": lm_output}]
        assert instance_detail["llm_score_output"] == lm_output
