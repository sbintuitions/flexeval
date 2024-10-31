from __future__ import annotations

import os
from typing import Any

import evaluate

from flexeval.core.metric.base import Metric, MetricResult
from flexeval.core.metric.string_processor import StringProcessor
from flexeval.core.utils.jinja2_utils import JINJA2_ENV

# by default, the program is not allowed to execute code and we need to set this environment variable
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class CodeEval(Metric):
    """
    A metric that evaluates generated code with test cases.

    Args:
        code_template: A Jinja2 template string to make the generated code.
            The template can contain variables from task inputs.
            If `None`, the code prompt will be the generated text itself.
        processor: A processor applied to model outputs before evaluation.
        evaluate_module: An evaluate module to use.

    Examples:
        >>> from flexeval import CodeEval
        >>> code_eval = CodeEval()
        >>> lm_outputs = ["def add(a, b):\\n    return a + b", "def is_equal(a, b):\\n    return a = b"]
        >>> references_list = [["assert add(1, 2) == 3"], ["assert is_equal(1, 2) == False"]]
        >>> result = code_eval.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={'pass@1': 0.5},
            instance_details=[
                {'passed': True, 'result': 'passed'},
                {'passed': False, 'result': 'failed: invalid syntax (<string>, line 2)'}
            ]
        )
    """

    def __init__(
        self,
        code_template: str | None = None,
        processor: StringProcessor | None = None,
        evaluate_module: str = "code_eval",
    ) -> None:
        if code_template is None:
            code_template = "{{ lm_output }}"

        self.code_template = JINJA2_ENV.from_string(code_template)
        self.code_eval = evaluate.load(evaluate_module)
        self.processor = processor

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if task_inputs_list is None:
            task_inputs_list = [{} for _ in lm_outputs]

        generated_code_list: list[str] = []
        test_case_list: list[str] = []
        # in code generation tasks, references_list contains the test cases
        for lm_output, task_inputs, test_cases in zip(
            lm_outputs,
            task_inputs_list,
            references_list,
        ):
            if self.processor is not None:
                lm_output = self.processor(lm_output)  # noqa: PLW2901

            generated_code = self.code_template.render(lm_output=lm_output, **task_inputs)

            generated_code_list.append(generated_code)
            test_case_list.append("\n".join(test_cases))
        pass_at_k, results = self.code_eval.compute(
            references=test_case_list,
            predictions=[[c] for c in generated_code_list],
            k=[1],
        )

        # `results` contain the detailed results for each test case
        # e.g., {0: [(0, {'task_id': 0, 'passed': False, 'result': "failed", 'completion_id': 0})]}
        results: dict[int, list[tuple[int, dict[str, Any]]]]

        instance_details: list[dict[str, Any]] = []
        for i in range(len(lm_outputs)):
            first_result = results[i][0]  # we only assume one candidate code per instance, so we take the first result
            _, detail_result = first_result  # the first element is just the index so we ignore it
            # remove unnecessary fields to save space
            detail_result.pop("completion_id")
            detail_result.pop("task_id")
            instance_details.append(detail_result)

        return MetricResult(pass_at_k, instance_details=instance_details)
