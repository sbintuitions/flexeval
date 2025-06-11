import pytest

from flexeval.core.metric import CodeEval
from flexeval.core.string_processor import RegexExtractor, StringProcessor


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
    ("code", "test_case", "lm_output_processor"),
    [
        (
            "```python\ndef add(a, b):\n    return a + b\n```",
            "assert add(1, 2) == 3",
            RegexExtractor("```python(.*?)```"),
        ),
    ],
)
def test_correct_code_with_processor(code: str, test_case: str, lm_output_processor: StringProcessor) -> None:
    code_eval = CodeEval(lm_output_processor=lm_output_processor)
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
    code_eval = CodeEval(code_template="{{ prompt }}{{ lm_output }}")
    metric_result = code_eval.evaluate([code], references_list=[[test_case]], extra_info_list=[{"prompt": prompt}])
    assert metric_result.summary == {"pass@1": 1.0}
    assert metric_result.instance_details[0]["passed"]
