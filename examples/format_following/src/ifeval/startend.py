# Copyright 2004 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from examples.format_following.src.base import ResponseConstraint


class EndChecker(ResponseConstraint):
    """Checks that the response ends with a specific phrase."""

    def __init__(self, end_phrase: str) -> None:
        """
        Args:
            end_phrase: A string representing the exact phrase the response should end with.
        """
        if not end_phrase or not isinstance(end_phrase, str):
            msg = "end_phrase must be a non-empty string."
            raise ValueError(msg)
        self.end_phrase = end_phrase.strip()

    def check(self, response: str) -> bool:
        if not isinstance(response, str):
            msg = "Response must be a string."
            raise TypeError(msg)

        response_cleaned = response.strip().strip('"').lower()
        end_phrase_cleaned = self.end_phrase.lower()
        return response_cleaned.endswith(end_phrase_cleaned)


class Quotation(ResponseConstraint):
    """Checks if the response is wrapped with double quotation marks."""

    def check(self, response: str) -> bool:
        response = response.strip()
        return len(response) > 1 and response[0] == '"' and response[-1] == '"'
