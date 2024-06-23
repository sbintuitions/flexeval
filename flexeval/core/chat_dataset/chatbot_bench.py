from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

from .base import ChatDataset, ChatInstance


def resolve_path_or_name(path_or_name: str | PathLike[str]) -> Path:
    if not Path(path_or_name).exists():
        curated_dataset_path = Path(__file__).parent / "chatbot_bench_datasets" / f"{path_or_name}.jsonl"
        if not curated_dataset_path.exists():
            msg = f"Unknown dataset path or name: {path_or_name}"
            raise ValueError(msg)
        file_path = curated_dataset_path
    else:
        file_path = path_or_name
    return file_path


class ChatbotBench(ChatDataset):
    """This class loads data with the jsonl format used in chat evaluation benchmarks such as
    MT-Bench (Multi-turn Benchmark) or Vicuna QA Benchmark.

    Example of a line from a jsonl file:
        {
          "question_id": 00,
          "category": "writing",
          "turns": [
            "Compose an engaging travel blog post about a recent trip to Hawaii.",
            "Rewrite your previous response. Start every sentence with the letter A."
          ]
        }
    """

    def __init__(
        self,
        path_or_name: str,
        ref_path_or_name: str | None = None,
        need_ref_categories: list[str] | None = None,
    ) -> None:
        file_path = resolve_path_or_name(path_or_name)

        self._id_to_question_id: list[int | str] = []
        self._id_to_category: list[str] = []
        self._messages_dict: dict[int | str, list[dict[str, str]]] = {}
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                self._id_to_question_id.append(item["question_id"])
                self._id_to_category.append(item["category"])
                self._messages_dict[item["question_id"]] = [{"role": "user", "content": turn} for turn in item["turns"]]

        self._references_dict: dict[int | str, list[str]] = {}
        if ref_path_or_name is not None:
            ref_file_path = resolve_path_or_name(ref_path_or_name)
            with open(ref_file_path) as f:
                for line in f:
                    item = json.loads(line)
                    self._references_dict[item["question_id"]] = item["choices"][0]["turns"]

        self.need_ref_categories = need_ref_categories or [
            "math",
            "coding",
            "reasoning",
        ]

    def require_incremental_response(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._id_to_question_id)

    def __getitem__(self, i: int) -> ChatInstance:
        question_id = self._id_to_question_id[i]
        category = self._id_to_category[i]
        references: list[str] = []
        if category in self.need_ref_categories:
            references = self._references_dict.get(question_id, [])
        return ChatInstance(self._messages_dict[question_id], references=references, extra_info={"category": category})
