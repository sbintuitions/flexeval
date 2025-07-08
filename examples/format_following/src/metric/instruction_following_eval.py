from __future__ import annotations

from collections import defaultdict
from typing import Any, TypeVar

import jsonargparse

from examples.format_following.src.base import ResponseConstraint
from flexeval import Metric, MetricResult

Module = TypeVar("Module", bound=Any)


class FormatFollowingMetric(Metric):
    """Metric for evaluating instruction following tasks.

    The task inputs should contain a list of constraints in the "constraints" field, each of which is an initialization
    parameter for InstructionFollowingChecker.
    """

    @staticmethod
    def _instantiate_checker_from_params(
        params: dict[str, str],
    ) -> ResponseConstraint:
        parser = jsonargparse.ArgumentParser(parser_mode="jsonnet")
        parser.add_argument("--module", type=ResponseConstraint, required=True)
        args = jsonargparse.Namespace(module=params)
        instantiated_config = parser.instantiate_classes(args)
        return instantiated_config.module

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        is_check_passed_list: list[list[bool]] = []
        is_check_passed_per_checker = defaultdict(list)
        for lm_output, extra_info in zip(lm_outputs, extra_info_list):
            constraints = [self._instantiate_checker_from_params(params) for params in extra_info["constraints"]]
            is_check_passed = [checker.check(lm_output) for checker in constraints]
            is_check_passed_list.append(is_check_passed)
            for checker, is_passed in zip(constraints, is_check_passed):
                is_check_passed_per_checker[checker.__class__.__name__].append(is_passed)

        num_items = len(is_check_passed_list)
        num_items_passed = sum(all(is_check_passed) for is_check_passed in is_check_passed_list)
        macro_accuracy = num_items_passed / num_items

        total_num_checks = sum(len(is_check_passed) for is_check_passed in is_check_passed_list)
        num_checks_passed = sum(sum(is_check_passed) for is_check_passed in is_check_passed_list)
        micro_accuracy = num_checks_passed / total_num_checks

        return MetricResult(
            summary={
                "format_accuracy-macro": macro_accuracy,
                "format_accuracy-micro": micro_accuracy,
                **{
                    f"format_accuracy-{checker_name}": sum(is_passed) / len(is_passed)
                    for checker_name, is_passed in is_check_passed_per_checker.items()
                },
            },
            instance_details=[
                {
                    "is_check_passed": is_check_passed,
                }
                for is_check_passed in is_check_passed_list
            ],
        )
