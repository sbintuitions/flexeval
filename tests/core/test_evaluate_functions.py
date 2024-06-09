from __future__ import annotations

import json
import tempfile

import pytest

from flexeval.core.evaluate_chat_response import evaluate_chat_response
from flexeval.core.evaluate_from_file import evaluate_from_file
from flexeval.core.evaluate_generation import evaluate_generation
from flexeval.core.evaluate_multiple_choice import evaluate_multiple_choice
from flexeval.core.evaluate_pairwise import Match, evaluate_pairwise
from flexeval.core.evaluate_perplexity import evaluate_perplexity
from flexeval.core.metric import ExactMatch
from flexeval.core.prompt_template import Jinja2PromptTemplate
from tests.dummy_modules import (
    DummyChatDataset,
    DummyGenerationDataset,
    DummyLanguageModel,
    DummyMultipleChoiceDataset,
    DummyPairwiseJudge,
    DummyTextDataset,
)


@pytest.mark.parametrize("require_incremental_response", [True, False])
def test_evaluate_chat_response(require_incremental_response: bool) -> None:
    metrics, outputs = evaluate_chat_response(
        language_model=DummyLanguageModel(),
        gen_kwargs={},
        eval_dataset=DummyChatDataset(
            require_incremental_response=require_incremental_response,
        ),
        metrics=[],
        batch_size=1,
    )
    assert isinstance(metrics, dict)
    assert isinstance(outputs, list)


def test_evaluate_generation() -> None:
    metrics, outputs = evaluate_generation(
        language_model=DummyLanguageModel(),
        gen_kwargs={},
        eval_dataset=DummyGenerationDataset(),
        prompt_template=Jinja2PromptTemplate("{{text}}"),
        metrics=[ExactMatch()],
        batch_size=1,
    )
    assert isinstance(metrics, dict)
    assert isinstance(outputs, list)


def test_evaluate_multiple_choice() -> None:
    metrics, outputs = evaluate_multiple_choice(
        language_model=DummyLanguageModel(),
        eval_dataset=DummyMultipleChoiceDataset(),
        prompt_template=Jinja2PromptTemplate("{{text}}"),
        batch_size=1,
    )
    assert isinstance(metrics, dict)
    assert isinstance(outputs, list)


def test_evaluate_perplexity() -> None:
    metrics = evaluate_perplexity(
        language_model=DummyLanguageModel(),
        eval_dataset=DummyTextDataset(),
        batch_size=1,
    )
    assert isinstance(metrics, dict)


def test_evaluate_from_file() -> None:
    items = [
        {"lm_output": "This is test", "references": "This is test"},
        {"lm_output": "This is test", "references": "This is not test"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for output_item in items:
            f.write(json.dumps(output_item) + "\n")
        f.flush()

        metrics_summary_dict, instance_metrics_list = evaluate_from_file(
            eval_file=f.name,
            metrics=[ExactMatch()],
        )

        assert metrics_summary_dict == {"exact_match": 0.5}
        assert instance_metrics_list == [{"exact_match": 1.0}, {"exact_match": 0.0}]


@pytest.mark.parametrize(
    "cached_matches",
    [
        None,
        [
            Match(
                model1="model1",
                model1_item={"lm_output": "dummy"},
                model2="model2",
                model2_item={"lm_output": "dummy"},
                winner="draw",
                rationale="cache test",
            ),
        ],
    ],
)
def test_evaluate_pairwise(cached_matches: list[Match] | None) -> None:
    num_items = 2
    model_scores_dict, match_info_list = evaluate_pairwise(
        model_items={
            "model1": [{"lm_output": "dummy"} for _ in range(num_items)],
            "model2": [{"lm_output": "dummy"} for _ in range(num_items)],
        },
        judge=DummyPairwiseJudge(),
        cached_matches=cached_matches,
    )
    assert model_scores_dict.keys() == {"win_rate", "bradley_terry"}
    # By default, match is made by AllCombinations.
    # It makes match by switching the order of model1 and model2.
    assert len(match_info_list) == 2 * num_items
    if cached_matches:
        assert match_info_list[0]["rationale"] == "cache test"
