import sacrebleu

from .base import GenerationDataset, GenerationInstance


class SacreBleuDataset(GenerationDataset):
    """Load datasets from the [sacrebleu](https://github.com/mjpost/sacrebleu) library.
    The available datasets are defined in sacrebleu.DATASETS.
    """

    def __init__(self, name: str, langpair: str) -> None:
        self._source_list: list[str] = list(sacrebleu.DATASETS[name].source(langpair))
        self._references_list: list[list[str]] = [
            [r.strip() for r in refs] for refs in sacrebleu.DATASETS[name].references(langpair)
        ]

        if len(self._source_list) != len(self._references_list):
            msg = "The number of source and reference pairs should be the same."
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self._source_list)

    def __getitem__(self, i: int) -> GenerationInstance:
        return GenerationInstance(
            inputs={"source": self._source_list[i]},
            references=self._references_list[i],
        )
