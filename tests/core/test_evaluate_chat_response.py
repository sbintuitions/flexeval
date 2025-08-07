from __future__ import annotations

import itertools

import pytest

from flexeval.core.evaluate_chat_response import evaluate_chat_response
from flexeval.core.few_shot_generator import RandomFewShotGenerator
from tests.dummy_modules import (
    DummyChatDataset,
    DummyLanguageModel,
)


@pytest.mark.parametrize(
    ("use_few_shot", "max_instances", "use_tools", "batch_size"),
    list(itertools.product([True, False], [None, 1], [True, False], [1, 3])),
)
def test_evaluate_chat_response(use_few_shot: bool, max_instances: int, use_tools: bool, batch_size: int) -> None:
    few_shot_generator = None
    if use_few_shot:
        few_shot_generator = RandomFewShotGenerator(dataset=DummyChatDataset(), num_shots=1, num_trials_to_avoid_leak=0)

    metrics, outputs = evaluate_chat_response(
        language_model=DummyLanguageModel(),
        gen_kwargs={},
        eval_dataset=DummyChatDataset(
            use_tools=use_tools,
        ),
        few_shot_generator=few_shot_generator,
        metrics=[],
        batch_size=batch_size,
        max_instances=max_instances,
    )
    assert isinstance(metrics, dict)
    assert metrics["finish_reason_ratio-length"] == 1.0
    assert isinstance(outputs, list)

    if max_instances is not None:
        assert len(outputs) <= max_instances

    # If the system message in "messages", few-shot examples should be inserted after the system message.
    # Therefore, in any case the system message should be in the first turn.
    assert outputs[0]["extra_info"]["messages"][0]["role"] == "system"

    if use_tools:
        assert isinstance(outputs[0]["extra_info"]["tool_calls"], list)
        assert isinstance(outputs[0]["extra_info"]["tools"], list)
        assert metrics["tool_call_validation_result_ratio-CompleteToolCall"] == 1.0
    else:
        assert "tool_calls" not in outputs[0]["extra_info"]
        assert "tools" not in outputs[0]["extra_info"]
        assert metrics["tool_call_validation_result_ratio-TextOnly"] == 1.0
