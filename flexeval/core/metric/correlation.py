from __future__ import annotations

import functools
import warnings
from typing import Literal

from scipy.stats import kendalltau, pearsonr, spearmanr

from flexeval.core.string_processor import StringProcessor

from .base import Metric, MetricResult


class Correlation(Metric):
    """
    Correlation metric to compute Pearson, Spearman, or Kendall correlation coefficients.
    The lm_outputs and references should be numeric values, optionally preprocessed by StringProcessor.

    Args:
        method: The correlation method to use ('pearson', 'spearman', 'kendall').
        lm_output_processor: StringProcessor or a list of StringProcessor to be applied to the model outputs before
            computing the correlation. If a list is provided, the processors will be applied in order.
        reference_processor: StringProcessor or a list of StringProcessor to be applied to the references before
            computing the correlation. If a list is provided, the processors will be applied in order.

    Examples:
        >>> from flexeval import Correlation
        >>> correlation = Correlation(method='pearson')
        >>> lm_outputs = ["1", "2", "3", "4", "5"]
        >>> references = [["5"], ["4"], ["3"], ["2"], ["1"]]
        >>> result = correlation.evaluate(lm_outputs, references)
        >>> print(result)
        MetricResult(
            summary={"pearson_correlation": -1.0, "pearson_pvalue": 0.0},
            instance_details=[],
        )
    """

    def __init__(
        self,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        lm_output_processor: StringProcessor | list[StringProcessor] | None = None,
        reference_processor: StringProcessor | list[StringProcessor] | None = None,
    ) -> None:
        if method not in {"pearson", "spearman", "kendall"}:
            msg = f"Invalid method '{method}'. Choose from 'pearson', 'spearman', 'kendall'."
            raise ValueError(msg)
        self.method = method

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
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        if len(lm_outputs) != len(references_list):
            msg = (
                f"Number of model outputs ({len(lm_outputs)}) and number of references ({len(references_list)}) "
                "should be the same."
            )
            raise ValueError(msg)

        # We only use the first reference here
        references = [refs[0] for refs in references_list]

        if self.lm_output_processors:
            lm_outputs = [
                functools.reduce(lambda x, norm: norm(x), self.lm_output_processors, output) for output in lm_outputs
            ]

        if self.reference_processors:
            references = [
                functools.reduce(lambda x, norm: norm(x), self.reference_processors, ref) for ref in references
            ]

        # The model output should be converted to float, if fails it will be treated as 0
        lm_outputs_as_float: list[float] = []
        for output in lm_outputs:
            try:
                lm_outputs_as_float.append(float(output))
            except ValueError:  # noqa:PERF203
                warnings.warn(f"Failed to convert model output '{output}' to float. Treating it as 0.", stacklevel=2)
                lm_outputs_as_float.append(0.0)

        # The reference should be converted to float
        references_as_float = [float(ref) for ref in references]

        # Compute correlation
        if self.method == "pearson":
            correlation, pvalue = pearsonr(lm_outputs_as_float, references_as_float)
        elif self.method == "spearman":
            correlation, pvalue = spearmanr(lm_outputs_as_float, references_as_float)
        elif self.method == "kendall":
            correlation, pvalue = kendalltau(lm_outputs_as_float, references_as_float)
        else:
            msg = f"Unsupported method: {self.method}"
            raise ValueError(msg)

        return MetricResult(
            {f"{self.method}_correlation": correlation, f"{self.method}_pvalue": pvalue},
            instance_details=[],
        )
