import zipfile
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from flexeval import GenerationDataset, GenerationInstance


class JMMLU(GenerationDataset):
    """Load the JMMLU dataset."""

    def __init__(self) -> None:
        # Download "https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/main/JMMLU.zip"
        dataset_file_path = Path(__file__).parent / "JMMLU.zip"
        if not dataset_file_path.exists():
            logger.info("Downloading JMMLU dataset...")
            response = requests.get(
                "https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/main/JMMLU.zip",
                timeout=10,
            )
            dataset_file_path.write_bytes(response.content)
            logger.info("Downloaded JMMLU dataset.")

        if not (dataset_file_path.parent / "JMMLU").exists():
            with zipfile.ZipFile(dataset_file_path, "r") as zip_ref:
                zip_ref.extractall(dataset_file_path.parent)

        # load JMMLU data
        instances: list[GenerationInstance] = []
        for file_path in (Path(__file__).parent / "JMMLU" / "test").glob("*.csv"):
            category_name = file_path.stem
            data_frame = pd.read_csv(file_path, encoding="utf-8-sig")
            for _i, item in enumerate(data_frame.to_dict(orient="records")):
                item["subject"] = category_name
                instance = GenerationInstance(inputs=item, references=[item[item["answer"]]])
                instances.append(instance)
            self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int) -> GenerationInstance:
        return self.instances[i]
