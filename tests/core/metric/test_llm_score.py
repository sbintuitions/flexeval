from __future__ import annotations

import pytest

from flexeval import Jinja2PromptTemplate, LanguageModel
from flexeval.core.language_model.base import LMOutput
from flexeval.core.metric.llm_score import (
    ChatLLMScore,
    LLMScore,
    parse_score_from_evaluator_output,
    prepare_chat_input_for_evaluator,
)


class EchoBackLanguageModel(LanguageModel):
    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        return [LMOutput(text=text, finish_reason="length") for text in text_list]

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[LMOutput]:
        return [
            LMOutput(text=chat_messages[-1]["content"], finish_reason="length") for chat_messages in chat_messages_list
        ]


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


@pytest.mark.parametrize(
    ("lm_outputs", "task_inputs_list", "expected_summary"),
    [
        (
            ["This score is 1.", "This score is 2.", "This is a good one."],
            [
                {"category": "category-0"},
                {"category": "category-0"},
                {"category": "category-1"},
            ],
            {
                "llm_score": 1.5,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
            }
        ),
        (
            ["This score is 1.", "This score is 2.", "This score is 3.", "This is a good one."],
            [
                {"category": ["category-0"]},
                {"category": ["category-0", "category-1"]},
                {"category": []},
                {"category": [""]},
            ],
            {
                "llm_score": 2.0,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
                "llm_score/category-1": 2.0,
            }
        )
    ],
)
def test_llm_score_with_category(
    lm_outputs: list[str],
    task_inputs_list: list[dict[str, str | list[str]]],
    expected_summary: dict[str, float]
) -> None:
    metric = LLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        category_key="category",
    )
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        task_inputs_list=task_inputs_list,
    )

    assert metric_output.summary == expected_summary

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


@pytest.mark.parametrize(
    ("lm_outputs", "task_inputs_list", "expected_summary"),
    [
        (
            ["This score is 1.", "This score is 2.", "This is a good one."],
            [
                {"category": "category-0"},
                {"category": "category-0"},
                {"category": "category-1"},
            ],
            {
                "llm_score": 1.5,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
            }
        ),
        (
            ["This score is 1.", "This score is 2.", "This score is 3.", "This is a good one."],
            [
                {"category": ["category-0"]},
                {"category": ["category-0", "category-1"]},
                {"category": []},
                {"category": [""]},
            ],
            {
                "llm_score": 2.0,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
                "llm_score/category-1": 2.0,
            }
        )
    ],
)
def test_chat_llm_score_with_category(
    lm_outputs: list[str],
    task_inputs_list: list[dict[str, str | list[str]]],
    expected_summary: dict[str, float]
) -> None:
    metric = ChatLLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        category_key="category",
    )
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        task_inputs_list=task_inputs_list,
    )

    assert metric_output.summary == expected_summary

    for lm_output, instance_detail in zip(lm_outputs, metric_output.instance_details):
        assert instance_detail["llm_score_input"] == [{"role": "user", "content": lm_output}]
        assert instance_detail["llm_score_output"] == lm_output


def test_prepare_chat_input_for_evaluator() -> None:
    lm_outputs = ["Output1", "Output2"]
    references_list = [
        ["Reference1"],
        [],
    ]
    task_inputs_list = [
        {"messages": [{"role": "user", "content": "Input1"}]},
        {"messages": [{"role": "user", "content": "Input2"}]},
    ]
    prompt_template = Jinja2PromptTemplate(
        "{{ messages[0]['content'] }}, {{ lm_output }}{%- if references|length > 0 -%}, {{ references[0] }}{%- endif -%}"  # noqa: E501
    )
    system_messsage = Jinja2PromptTemplate(
        "{%- if references|length > 0 -%}With Reference{%- else -%}Without Reference{%- endif -%}"
    )

    evaluator_input_list = prepare_chat_input_for_evaluator(
        lm_outputs, references_list, task_inputs_list, prompt_template, system_messsage
    )

    assert evaluator_input_list[0][0] == {"role": "system", "content": "With Reference"}
    assert evaluator_input_list[0][1] == {"role": "user", "content": "Input1, Output1, Reference1"}

    assert evaluator_input_list[1][0] == {"role": "system", "content": "Without Reference"}
    assert evaluator_input_list[1][1] == {"role": "user", "content": "Input2, Output2"}
