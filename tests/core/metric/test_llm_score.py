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
from flexeval.core.metric.utils import extract_text_from_outputs


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
        return [
            LMOutput(text=chat_messages[-1]["content"], finish_reason="length") for chat_messages in chat_messages_list
        ]


@pytest.mark.parametrize(
    ("evaluator_output", "valid_score_range", "regex_to_parse_score", "expected_score"),
    [
        ("The final score is 65.", None, r"(\d+)", 65),
        ("The final score is 65.", (0, 5), r"(\d+)", None),
        ("Yes, this is a good one.", None, r"(\d+)", None),
        ("The score is 5. No, it is 6.", None, r"(\d+)", 6),
        ("The score is [[5]]. No, it is 6.", None, r"\[\[(\d+)\]\]", 5),
    ],
)
def test_parse_score_from_evaluator_output(
    evaluator_output: str,
    valid_score_range: tuple[int, int] | None,
    regex_to_parse_score: str,
    expected_score: int,
) -> None:
    score = parse_score_from_evaluator_output(
        evaluator_output, valid_score_range, regex_to_parse_score=regex_to_parse_score
    )
    assert score == expected_score


@pytest.mark.parametrize(
    ("lm_outputs", "extra_info_list", "regex_to_parse_score", "expected_summary"),
    [
        (
            ["This score is 1.", "This score is 2.", "This is a good one."],
            [
                {"category": "category-0"},
                {"category": "category-0"},
                {"category": "category-1"},
            ],
            r"(\d+)",
            {
                "llm_score": 1.5,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
            },
        ),
        (
            ["This score is 1.", "This score is 2.", "This score is 3.", "This is a good one."],
            [
                {"category": ["category-0"]},
                {"category": ["category-0", "category-1"]},
                {"category": []},
                {"category": [""]},
            ],
            r"(\d+)",
            {
                "llm_score": 2.0,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
                "llm_score/category-1": 2.0,
            },
        ),
        (
            ["This score is [[1]].", "This score is 2.", "This score is 3.", "This is a good one."],
            [
                {"category": ["category-0"]},
                {"category": ["category-0", "category-1"]},
                {"category": []},
                {"category": [""]},
            ],
            r"\[\[(\d+)\]\]",
            {
                "llm_score": 1.0,
                "num_failed_score_parses": 3,
                "llm_score/category-0": 1.0,
            },
        ),
    ],
    indirect=["lm_outputs"],
)
@pytest.mark.parametrize("metric_prefix", ["", "prefix"])
def test_llm_score(
    lm_outputs: list[str | LMOutput],
    extra_info_list: list[dict[str, str | list[str]]],
    expected_summary: dict[str, float],
    regex_to_parse_score: str,
    metric_prefix: str | None,
) -> None:
    metric = LLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        category_key="category",
        metric_prefix=metric_prefix,
        regex_to_parse_score=regex_to_parse_score
    )
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )
    if metric_prefix:
        metric_prefix += "-"

    assert metric_output.summary == {f"{metric_prefix}{k}": v for k, v in expected_summary.items()}

    for lm_output, instance_detail in zip(extract_text_from_outputs(lm_outputs), metric_output.instance_details):
        assert instance_detail[f"{metric_prefix}llm_score_input"] == lm_output
        assert instance_detail[f"{metric_prefix}llm_score_output"] == lm_output


@pytest.mark.parametrize(
    ("lm_outputs", "extra_info_list", "regex_to_parse_score", "expected_summary"),
    [
        (
            ["This score is 1.", "This score is 2.", "This is a good one."],
            [
                {"category": "category-0"},
                {"category": "category-0"},
                {"category": "category-1"},
            ],
            r"(\d+)",
            {
                "llm_score": 1.5,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
            },
        ),
        (
            ["This score is 1.", "This score is 2.", "This score is 3.", "This is a good one."],
            [
                {"category": ["category-0"]},
                {"category": ["category-0", "category-1"]},
                {"category": []},
                {"category": [""]},
            ],
            r"(\d+)",
            {
                "llm_score": 2.0,
                "num_failed_score_parses": 1,
                "llm_score/category-0": 1.5,
                "llm_score/category-1": 2.0,
            },
        ),
        (
            ["This score is [[1]].", "This score is 2.", "This score is 3.", "This is a good one."],
            [
                {"category": ["category-0"]},
                {"category": ["category-0", "category-1"]},
                {"category": []},
                {"category": [""]},
            ],
            r"\[\[(\d+)\]\]",
            {
                "llm_score": 1.0,
                "num_failed_score_parses": 3,
                "llm_score/category-0": 1.0,
            },
        ),
    ],
    indirect=["lm_outputs"],
)
@pytest.mark.parametrize("metric_prefix", ["", "prefix"])
def test_chat_llm_score(
    lm_outputs: list[str | LMOutput],
    extra_info_list: list[dict[str, str | list[str]]],
    expected_summary: dict[str, float],
    regex_to_parse_score: str,
    metric_prefix: str,
) -> None:
    metric = ChatLLMScore(
        language_model=EchoBackLanguageModel(),
        prompt_template=Jinja2PromptTemplate("{{ lm_output }}"),
        category_key="category",
        metric_prefix=metric_prefix,
        regex_to_parse_score=regex_to_parse_score
    )
    metric_output = metric.evaluate(
        lm_outputs=lm_outputs,
        extra_info_list=extra_info_list,
    )

    if metric_prefix:
        metric_prefix += "-"
    assert metric_output.summary == {f"{metric_prefix}{k}": v for k, v in expected_summary.items()}

    for lm_output, instance_detail in zip(extract_text_from_outputs(lm_outputs), metric_output.instance_details):
        assert instance_detail[f"{metric_prefix}llm_score_input"] == [{"role": "user", "content": lm_output}]
        assert instance_detail[f"{metric_prefix}llm_score_output"] == lm_output


def test_prepare_chat_input_for_evaluator() -> None:
    lm_outputs = ["Output1", "Output2"]
    references_list = [
        ["Reference1"],
        [],
    ]
    extra_info_list = [
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
        lm_outputs, references_list, extra_info_list, prompt_template, system_messsage
    )

    assert evaluator_input_list[0][0] == {"role": "system", "content": "With Reference"}
    assert evaluator_input_list[0][1] == {"role": "user", "content": "Input1, Output1, Reference1"}

    assert evaluator_input_list[1][0] == {"role": "system", "content": "Without Reference"}
    assert evaluator_input_list[1][1] == {"role": "user", "content": "Input2, Output2"}
