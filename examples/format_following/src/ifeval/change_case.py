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

import warnings
from typing import Literal

import langdetect
import nltk

from examples.format_following.src.base import ResponseConstraint


class CapitalWordFrequency(ResponseConstraint):
    """
    Checks frequency of words with all capital letters.

    Args:
      capital_frequency: An integer that represents the number of words that
        should be in all capital letters.
      capital_relation: A string that is 'at least' or 'at most' that refers to
        the frequency.
    """

    def __init__(self, capital_relation: Literal["less than", "at least"], capital_frequency: int) -> None:
        self.capital_relation = capital_relation
        self.capital_frequency = capital_frequency

        nltk.download("punkt_tab", quiet=True)

    def check(self, response: str) -> bool:
        words = nltk.word_tokenize(response)
        num_capital_words = sum(word.isupper() for word in words)

        if self.capital_relation == "less than":
            return num_capital_words < self.capital_frequency
        if self.capital_relation == "at least":
            return num_capital_words >= self.capital_frequency

        msg = f"Invalid capital_relation: {self.capital_relation}"
        raise ValueError(msg)


class EnglishCapital(ResponseConstraint):
    """Checks that the response is in english and is in all capital letters."""

    def check(self, response: str) -> bool:
        try:
            return response.isupper() and langdetect.detect(response) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            warnings.warn(f"Unable to detect language for text {response} due to {e}", stacklevel=2)
            return True


class EnglishLowercase(ResponseConstraint):
    """Checks that the response is in english and is in all lowercase letters."""

    def check(self, response: str) -> bool:
        try:
            return response.islower() and langdetect.detect(response) == "en"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            warnings.warn(f"Unable to detect language for text {response} due to {e}", stacklevel=2)
            return True
