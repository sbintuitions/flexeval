from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from loguru import logger

from .base import Metric, MetricResult

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")


def retry_on_error(
    perspectiveapi_call: Callable,
    max_num_trials: int = 8,
    first_wait_time: int = 1,
) -> dict[str, Any] | None:
    """Retry the function call when an error occurs."""
    time.sleep(first_wait_time)
    for i in range(max_num_trials):
        try:
            return perspectiveapi_call()
        except HttpError as e:  # noqa: PERF203
            if i == max_num_trials - 1:
                raise
            logger.warning(f"We got an error: {e}")
            wait_time_seconds = first_wait_time * (2**i)
            logger.warning(f"Wait for {wait_time_seconds} seconds...")
            time.sleep(wait_time_seconds)
    return None


class PerspectiveAPI(Metric):
    """A metric that evaluates text outputs using the Perspective API.
    Please set `PERSPECTIVE_API_KEY` in the environment variable.

    Args:
        languages: A list of languages to analyze.

    Examples:
        >>> from flexeval import PerspectiveAPI
        >>> perspective_api = PerspectiveAPI(languages=["en"])
        >>> lm_outputs = ["I love you", "I hate you"]
        >>> result = perspective_api.evaluate(lm_outputs)
        >>> print(result)
        MetricResult(
            summary={'TOXICITY': 0.35407552, ..., 'THREAT': 0.0265799825},
            instance_details=[
                {'TOXICITY': 0.02543884, ..., 'THREAT': 0.009204263},
                {'TOXICITY': 0.6827122, ..., 'THREAT': 0.043955702}
                ]
            )
    """

    def __init__(self, languages: list[str]) -> None:
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PERSPECTIVE_API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        self.languages = languages
        self.attributes = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"]

    def evaluate(
        self,
        lm_outputs: list[str],
        references_list: list[list[str]] | None = None,
        task_inputs_list: list[dict[str, str]] | None = None,
    ) -> MetricResult:
        instance_details = []
        for lm_output in lm_outputs:
            if lm_output == "":
                instance_details.append({att: 0.0 for att in self.attributes})
                continue
            analyze_request = {
                "comment": {"text": lm_output},
                "languages": self.languages,
                "requestedAttributes": {att: {} for att in self.attributes},
            }
            response = retry_on_error(perspectiveapi_call=self.client.comments().analyze(body=analyze_request).execute)
            instance_details.append(
                {att: response["attributeScores"][att]["summaryScore"]["value"] for att in self.attributes},
            )
        scores_for_attribute = {att: [] for att in self.attributes}
        for instance in instance_details:
            for att in self.attributes:
                scores_for_attribute[att].append(instance[att])
        average_scores = {att: np.mean(scores_for_attribute[att]) for att in self.attributes}
        return MetricResult(average_scores, instance_details=instance_details)
