# Dummy QA dataset

This is a dummy QA dataset for testing purposes.
The dataset contains the save data as `tests/dummy_modules/test.jsonl`.
```json lines
{"id": 0, "question": "What is the highest mountain in the world.", "answers": ["Mount Everest", "Everest"]}
{"id": 1, "question": "What is the chemical symbol for water?", "answers": ["H2O"]}
{"id": 2, "question": "In which year did the Titanic sink?", "answers": ["1912"]}
{"id": 3, "question": "What is the capital of France?", "answers": ["Paris"]}
{"id": 4, "question": "Who wrote 'Romeo and Juliet'?", "answers": ["William Shakespeare", "Shakespeare"]}
{"id": 5, "question": "What is the largest planet in our solar system?", "answers": ["Jupiter"]}
{"id": 6, "question": "What is the process by which plants make their food?", "answers": ["Photosynthesis"]}
{"id": 7, "question": "Who painted the Mona Lisa?", "answers": ["Leonardo da Vinci", "da Vinci"]}
{"id": 8, "question": "What is the smallest prime number?", "answers": ["2"]}
{"id": 9, "question": "Who developed the theory of relativity?", "answers": ["Albert Einstein", "Einstein"]}
```

Generated from the following code.
```python
import datasets
import json

items = []
with open("tests/dummy_modules/test.jsonl") as f:
    for line in f:
        item = json.loads(line)
        items.append(item)

dataset = datasets.Dataset.from_list(items)
dataset.to_parquet("tests/dummy_modules/hf_dataset/train.parquet")
```
