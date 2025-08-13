# Copyright 2025 SBIntuitions Authors.
# Modifications were made by SBIntuitions in 2025.
#
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

import re
from typing import Literal

from examples.format_following.src.base import ResponseConstraint


class Existence(ResponseConstraint):
    """Checks the existence of certain keywords in the response."""

    def __init__(self, keywords: list[str]) -> None:
        """
        Args:
            keywords: A list of strings representing the keywords that are
                expected in the response.
        """
        if not keywords:
            msg = "A list of keywords must be provided."
            raise ValueError(msg)
        self.keywords = sorted(keywords)

    def check(self, response: str) -> bool:
        return all(re.search(re.escape(keyword), response, flags=re.IGNORECASE) for keyword in self.keywords)


class ForbiddenWords(ResponseConstraint):
    """Checks that specified words are not used in the response."""

    def __init__(self, forbidden_words: list[str]) -> None:
        """
        Args:
            forbidden_words: A list of strings representing words that are not
                allowed in the response.
        """
        if not forbidden_words:
            msg = "Forbidden words list cannot be empty."
            raise ValueError(msg)
        self.forbidden_words = sorted(set(forbidden_words))

    def check(self, response: str) -> bool:
        return all(not re.search("\\b" + word + "\\b", response, flags=re.IGNORECASE) for word in self.forbidden_words)


class Frequency(ResponseConstraint):
    """Checks the keyword frequency in the response."""

    def __init__(self, keyword: str, frequency: int, relation: Literal["less than", "at least"]) -> None:
        """
        Args:
            keyword: A string representing a keyword that is expected in the response.
            frequency: An integer specifying the required frequency of the keyword.
            relation: A string specifying the relational operator for comparison,
                either "less than" or "at least".
        """
        self.keyword = keyword.strip()
        self.frequency = max(1, frequency)  # Ensure frequency is at least 1.
        if relation not in {"less than", "at least"}:
            msg = f"Invalid relation: {relation}. Must be 'less than' or 'at least'."
            raise ValueError(msg)
        self.relation = relation

    def check(self, response: str) -> bool:
        actual_occurrences = len(re.findall(re.escape(self.keyword), response, flags=re.IGNORECASE))

        if self.relation == "less than":
            return actual_occurrences < self.frequency
        if self.relation == "at least":
            return actual_occurrences >= self.frequency
        return False


class LetterFrequency(ResponseConstraint):
    """Checks letter frequency."""

    def __init__(self, letter: str, let_frequency: int, let_relation: Literal["less than", "at least"]) -> None:
        """
        Args:
            letter: A string representing a letter that is expected in the response.
            let_frequency: An integer specifying the number of times the letter is
                expected to appear in the response.
            let_relation: A string specifying the relational operator for comparison,
                either "less than" or "at least".
        """
        if not letter or len(letter) != 1:
            msg = "letter must be a single character."
            raise ValueError(msg)

        self.letter = letter.lower()
        self.let_frequency = max(0, let_frequency)

        if let_relation not in ["less than", "at least"]:
            msg = "let_relation must be either 'less than' or 'at least'."
            raise ValueError(msg)

        self.let_relation = let_relation

    def check(self, response: str) -> bool:
        response = response.lower()
        letter_count = response.count(self.letter)

        if self.let_relation == "less than":
            return letter_count < self.let_frequency
        if self.let_relation == "at least":
            return letter_count >= self.let_frequency
        msg = f"Invalid relation: {self.let_relation}"
        raise ValueError(msg)
