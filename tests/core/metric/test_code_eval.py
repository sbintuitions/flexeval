import pytest

from flexeval.core.metric import CodeEval
from flexeval.core.metric.normalizer import Normalizer, RegexNormalizer


@pytest.mark.parametrize(
    ("code", "test_case"),
    [
        ("def add(a, b):\n    return a + b", "assert add(1, 2) == 3"),
        ("def subtract(a, b):\n    return a - b", "assert subtract(1, 2) == -1"),
        ("import math", "assert math.sqrt(4) == 2.0"),
    ],
)
def test_correct_code(code: str, test_case: str) -> None:
    code_eval = CodeEval()
    metric_result = code_eval.evaluate([code], references_list=[[test_case]])
    assert metric_result.summary == {"pass@1": 1.0}


@pytest.mark.parametrize(
    ("code", "test_case", "normalizer"),
    [
        (
            "```python\ndef add(a, b):\n    return a + b\n```",
            "assert add(1, 2) == 3",
            RegexNormalizer("```python(.*?)```"),
        ),
    ],
)
def test_correct_code_with_normalizer(code: str, test_case: str, normalizer: Normalizer) -> None:
    code_eval = CodeEval(normalizer=normalizer)
    metric_result = code_eval.evaluate([code], references_list=[[test_case]])
    assert metric_result.summary == {"pass@1": 1.0}


@pytest.mark.parametrize(
    ("code", "test_case"),
    [
        ("", "assert add(1, 2) == 3"),
        ("'asdfsad' + 100", "assert add(1, 2) == 3"),
        ("def add(a, b):\n    return a - b", "assert add(1, 2) == 3"),
    ],
)
def test_incorrect_code(code: str, test_case: str) -> None:
    code_eval = CodeEval()
    metric_result = code_eval.evaluate([code], references_list=[[test_case]])
    assert metric_result.summary == {"pass@1": 0.0}


@pytest.mark.parametrize(
    ("prompt", "code", "test_case"),
    [
        ("def add(a, b):", "\n    return a + b", "assert add(1, 2) == 3"),
        ("def subtract(a, b):", "\n    return a - b", "assert subtract(1, 2) == -1"),
    ],
)
def test_correct_code_with_prompt(prompt: str, code: str, test_case: str) -> None:
    code_eval = CodeEval(code_prompt_template="{{ prompt }}")
    metric_result = code_eval.evaluate([code], references_list=[[test_case]], task_inputs_list=[{"prompt": prompt}])
    assert metric_result.summary == {"pass@1": 1.0}
    assert metric_result.instance_details[0]["passed"]
