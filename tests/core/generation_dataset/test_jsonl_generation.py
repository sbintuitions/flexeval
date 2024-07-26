from __future__ import annotations

import json
import tempfile
from os import PathLike

import pytest

from flexeval.core.generation_dataset import JsonlGenerationDataset


@pytest.fixture()
def mock_jsonl_data_path() -> None:
    with tempfile.NamedTemporaryFile(mode="w") as f:
        for i in range(10):
            f.write(
                json.dumps({"input": f"test_input_{i}", "output": f"test_output_{i}"}) + "\n",
            )
        f.flush()
        yield f.name


def test_jsonl_generation_dataset(mock_jsonl_data_path: str | PathLike[str]) -> None:
    dataset = JsonlGenerationDataset(
        path=mock_jsonl_data_path,
        reference_template="{{ output }}",
    )

    assert len(dataset) > 0

    item = dataset[0]
    assert item.inputs == {
        "input": "test_input_0",
        "output": "test_output_0",
    }

    assert item.references == ["test_output_0"]


def test_data_range(mock_jsonl_data_path: str | PathLike[str]) -> None:
    dataset = JsonlGenerationDataset(
        path=mock_jsonl_data_path,
        reference_list_template="['{{ output }}']",
        data_range=(2, 5),
    )
    assert len(dataset) == 3
    for i, item in enumerate(dataset, start=2):
        assert item.inputs == {
            "input": f"test_input_{i}",
            "output": f"test_output_{i}",
        }

        assert item.references == [f"test_output_{i}"]
