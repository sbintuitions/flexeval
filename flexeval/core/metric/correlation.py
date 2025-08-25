from __future__ import annotations

import warnings
from typing import Literal

from scipy.stats import kendalltau, pearsonr, spearmanr

from flexeval.core.language_model.base import LMOutput
from flexeval.core.string_processor import StringProcessor

from .base import Metric, MetricResult
from .utils import apply_string_processors, extract_text_from_outputs, validate_inputs


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

        self.lm_output_processors = lm_output_processor
        self.reference_processors = reference_processor

    def evaluate(
        self,
        lm_outputs: list[str | LMOutput],
        references_list: list[list[str]],
        extra_info_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        validate_inputs(lm_outputs, references_list, extra_info_list)

        lm_outputs = extract_text_from_outputs(lm_outputs)
        lm_outputs = [apply_string_processors(output, self.lm_output_processors) for output in lm_outputs]

        references = [refs[0] for refs in references_list]
        references = [apply_string_processors(ref, self.reference_processors) for ref in references]

        # Convert to numeric values
        lm_outputs_as_float: list[float] = []
        for output in lm_outputs:
            try:
                lm_outputs_as_float.append(float(output))
            except ValueError:  # noqa:PERF203
                warnings.warn(f"Failed to convert model output '{output}' to float. Treating it as 0.", stacklevel=2)
                lm_outputs_as_float.append(0.0)

        references_as_float = [float(ref) for ref in references]

        # Compute metrics
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
