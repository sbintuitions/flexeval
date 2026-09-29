from __future__ import annotations

import json

import pytest

from flexeval.core.chat_dataset import ChatbotBench, ChatInstance
from flexeval.core.chat_dataset.chatbot_bench import resolve_path_or_name


@pytest.mark.parametrize(
    ("path", "ref_name"),
    [
        ("mt-en", "mt-en-ref-gpt4"),
        ("mt-ja", "mt-ja-ref-gpt4"),
        ("rakuda-v2-ja", None),
        ("vicuna-en", "vicuna-en-ref-gpt4"),
        ("vicuna-ja", "vicuna-ja-ref-gpt4"),
    ],
)
def test_chatbot_bench(path: str, ref_name: str | None) -> None:
    dataset = ChatbotBench(
        path_or_name=path,
        ref_path_or_name=ref_name,
        need_ref_categories=None,
    )

    assert len(dataset) > 0
    assert isinstance(dataset[0], ChatInstance)

    if ref_name is None:
        assert all(len(instance.references) == 0 for instance in dataset)
    else:
        assert any(len(instance.references) > 0 for instance in dataset)


def test_only_first_n() -> None:
    # mt-en (MT-Bench) should have multiple messages by default
    dataset = ChatbotBench(path_or_name="mt-en", load_only_first_n=1)
    assert all(len(instance.messages) == 1 for instance in dataset)


def test_messages_list_is_not_shared_between_instances() -> None:
    dataset = ChatbotBench(path_or_name="mt-en")
    original_messages = list(dataset[0].messages)

    dataset[0].messages.append({"role": "assistant", "content": "mutation"})

    assert dataset[0].messages == original_messages


def test_references_correspond_to_last_user_turn() -> None:
    dataset = ChatbotBench(
        path_or_name="mt-en",
        ref_path_or_name="mt-en-ref-gpt4",
        need_ref_categories=["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"],
    )
    with open(resolve_path_or_name("mt-en")) as f:
        questions = [json.loads(line) for line in f]
    with open(resolve_path_or_name("mt-en-ref-gpt4")) as f:
        references_by_id = {item["question_id"]: item["choices"][0]["turns"] for item in map(json.loads, f)}

    for instance, question in zip(dataset, questions):
        references_for_turns = references_by_id.get(question["question_id"], [])
        expected = [references_for_turns[len(question["turns"]) - 1]] if references_for_turns else []
        assert instance.references == expected
