from __future__ import annotations

from typing import Any

from flexeval.scripts.flexeval_file import EvalDataLoader


class DummyEvalDataLoader(EvalDataLoader):
    def load(self) -> list[dict[str, Any]]:
        return [
            {
                "task_inputs": {"input": f"dummy_input{i}"},
                "lm_output": f"dummy_output{i}",
                "references": [f"dummy_reference{i}"],
            }
            for i in range(4)
        ]
