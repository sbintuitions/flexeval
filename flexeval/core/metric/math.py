from __future__ import annotations

import functools
import warnings

import math_verify

from flexeval.core.string_processor.base import StringProcessor

from .base import Metric, MetricResult


class MathVerify(Metric):
    """
    The proportion of correct answers verified by `math-verify` https://github.com/huggingface/Math-Verify
    If there are multiple references, the output is considered correct if it matches any of the references.
    Args:
        lm_output_processor:
            StringProcessor or a list of StringProcessors to be applied to the model outputs before comparison.
        reference_processor: StringProcessor or list of Normalizers to apply to the references before comparison.
    Examples:
        >>> from flexeval.core.metric import MathVerify
        >>> math_verify = MathVerify()
        >>> lm_outputs = ["The answer is $a+b$", "The answer: $1$"]
        >>> references_list = [["Final answer: $b+a$"], ["A: $2$"]]
        >>> result = math_verify.evaluate(lm_outputs, references_list)
        >>> print(result)
        MetricResult(
            summary={"verified": 0.5},
            instance_details=[{"verified": True}, {"verified": False}],
        )
    """

    def __init__(
        self,
        lm_output_processor: StringProcessor | list[StringProcessor] | None = None,
        reference_processor: StringProcessor | list[StringProcessor] | None = None,
    ) -> None:
        if isinstance(lm_output_processor, StringProcessor):
            lm_output_processor = [lm_output_processor]
        if isinstance(reference_processor, StringProcessor):
            reference_processor = [reference_processor]

        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if len(lm_outputs) != len(references_list):
            msg = (
                f"Number of model outputs ({len(lm_outputs)}) and number of references ({len(references_list)}) "
                "should be the same."
            )
            raise ValueError(msg)

        if self.lm_output_processors:
            lm_outputs = [
                functools.reduce(lambda x, norm: norm(x), self.lm_output_processors, output) for output in lm_outputs
            ]

        if self.reference_processors:
            references_list = [
                [functools.reduce(lambda x, norm: norm(x), self.reference_processors, ref) for ref in references]
                for references in references_list
            ]

        verified_list: list[bool] = []
        extracted_answers: list[list] = []
        for lm_output, expected_outputs in zip(lm_outputs, references_list):
            is_correct = False
            answer = []
            for expected_output in expected_outputs:
                try:
                    expected = math_verify.parse(expected_output)
                    answer = math_verify.parse(lm_output)
                    verified = math_verify.verify(expected, answer)
                except Exception as e:  # noqa: BLE001
                    warnings.warn(f"Could not run math_verify on {lm_output} : {e}", stacklevel=1)
                    answer = ["n/a"]
                    verified = False
                is_correct |= verified
            verified_list.append(is_correct)
            extracted_answers.append(answer)

        return MetricResult(
            {"math_verify_accuracy": sum(verified_list) / len(verified_list)},
            instance_details=[
                {"math_verify_match": s, "extracted_answer": ans} for s, ans in zip(verified_list, extracted_answers)
            ],
        )
