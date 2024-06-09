from __future__ import annotations

from flexeval.core.generation_dataset import GenerationDataset, GenerationInstance


class DummyGenerationDataset(GenerationDataset):
    def __init__(self) -> None:
        self._data = [
            GenerationInstance(
                inputs={"text": "Hello, world!"},
                references=["Positive"],
            ),
            GenerationInstance(
                inputs={"text": "Good morning."},
                references=["Positive"],
            ),
            GenerationInstance(
                inputs={"text": "Good bye, world..."},
                references=["Negative"],
            ),
            GenerationInstance(
                inputs={"text": "Bad morning..."},
                references=["Negative"],
            ),
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int) -> GenerationInstance:
        return self._data[item]
