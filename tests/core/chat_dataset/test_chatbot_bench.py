from __future__ import annotations

import pytest

from flexeval.core.chat_dataset import ChatbotBench, ChatInstance


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
