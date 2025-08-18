from __future__ import annotations

import pytest

from flexeval.core.metric import CodeEval
from flexeval.core.string_processor import RegexExtractor, StringProcessor


@pytest.mark.parametrize(
    ("lm_outputs", "references", "lm_output_processor", "expected_score"),
    [
        (["def add(a, b):\n    return a + b"], ["assert add(1, 2) == 3"], None, 1.0),
        (["def subtract(a, b):\n    return a - b"], ["assert subtract(1, 2) == -1"], None, 1.0),
        (["import math"], ["assert math.sqrt(4) == 2.0"], None, 1.0),
        (
            ["```python\ndef add(a, b):\n    return a + b\n```"],
            ["assert add(1, 2) == 3"],
            RegexExtractor("```python(.*?)```"),
            1.0,
        ),
        # Incorrect test code
        ([""], ["assert add(1, 2) == 3"], None, 0.0),
        (["'asdfsad' + 100"], ["assert add(1, 2) == 3"], None, 0.0),
        (["def add(a, b):\n    return a - b"], ["assert add(1, 2) == 3"], None, 0.0),
    ],
    indirect=["lm_outputs"],
)
def test_code(
    lm_outputs: list[str], references: str, lm_output_processor: StringProcessor | None, expected_score: float
) -> None:
    code_eval = CodeEval(lm_output_processor=lm_output_processor)
    metric_result = code_eval.evaluate(lm_outputs, references_list=[[ref] for ref in references])
    assert metric_result.summary == {"pass@1": expected_score}


@pytest.mark.parametrize(
    ("prompt", "code", "test_case"),
    [
        ("def add(a, b):", "\n    return a + b", "assert add(1, 2) == 3"),
        ("def subtract(a, b):", "\n    return a - b", "assert subtract(1, 2) == -1"),
    ],
)
def test_correct_code_with_prompt(prompt: str, code: str, test_case: str) -> None:
    code_eval = CodeEval(code_template="{{ prompt }}{{ lm_output }}")
    metric_result = code_eval.evaluate([code], references_list=[[test_case]], extra_info_list=[{"prompt": prompt}])
    assert metric_result.summary == {"pass@1": 1.0}
    assert metric_result.instance_details[0]["passed"]
