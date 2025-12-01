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

import functools
import re
from typing import Literal

import nltk

from examples.format_following.src.base import ResponseConstraint


class NthParagraphFirstWord(ResponseConstraint):
    """Check the paragraph and the first word of the nth paragraph."""

    def __init__(self, num_paragraphs: int, nth_paragraph: int, first_word: str) -> None:
        """
        Args:
            num_paragraphs: An integer indicating the number of paragraphs expected in the response.
            nth_paragraph: An integer indicating the paragraph number to check (starting from 1).
            first_word: A string representing the first word of the nth paragraph.
        """
        self.num_paragraphs = max(1, num_paragraphs)
        self.nth_paragraph = max(1, nth_paragraph)
        self.first_word = first_word.lower()

    def check(self, response: str) -> bool:
        paragraphs = re.split(r"\n\n", response)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self.nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self.nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self.num_paragraphs and first_word == self.first_word


class NumberParagraphs(ResponseConstraint):
    """Checks the paragraphs in the response."""

    def __init__(self, num_paragraphs: int) -> None:
        """
        Args:
            num_paragraphs: An integer specifying the required number of paragraphs.
        """
        self.num_paragraphs = max(1, num_paragraphs)  # Ensure at least 1 paragraph.

    def check(self, response: str) -> bool:
        paragraphs = re.split(r"\s?\*\*\*\s?", response)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index in (0, len(paragraphs) - 1):
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self.num_paragraphs


class NumberSentences(ResponseConstraint):
    """Checks the number of sentences in the response."""

    def __init__(self, num_sentences: int, relation: Literal["less than", "at least"]) -> None:
        """
        Args:
            num_sentences: An integer specifying the number of sentences as a threshold.
            relation: A string in ("less than", "at least"), defining the relational operator for comparison.
                - "less than": The actual number of sentences < the threshold.
                - "at least": The actual number of sentences >= the threshold.
        """
        self.num_sentences = max(1, num_sentences)  # Ensure threshold is at least 1.
        if relation not in {"less than", "at least"}:
            msg = f"Invalid relation: {relation}. Must be 'less than' or 'at least'."
            raise ValueError(msg)
        self.relation = relation

    def check(self, response: str) -> bool:
        """Check if the number of sentences in the response follows the instruction.

        Args:
            response: A string representing the response.

        Returns:
            True if the response satisfies the sentence count condition; otherwise, False.
        """
        num_sentences = count_sentences(response)
        if self.relation == "less than":
            return num_sentences < self.num_sentences
        if self.relation == "at least":
            return num_sentences >= self.num_sentences
        return False


@functools.lru_cache
def _get_sentence_tokenizer() -> nltk.tokenize.punkt.PunktSentenceTokenizer:
    nltk.download("punkt_tab", quiet=True)
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def count_sentences(text: str) -> int:
    """Count the number of sentences."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)


class NumberWords(ResponseConstraint):
    """Checks the number of words in the response."""

    def __init__(self, relation: Literal["less than", "at least"], num_words: int) -> None:
        """
        Args:
            relation: A string specifying the relational operator for comparison.
                Supported values are "less than" and "at least".
            num_words: An integer specifying the number of words for the comparison.
        """
        if relation not in {"less than", "at least"}:
            msg = f"Unsupported relation: {relation}. Use 'less than' or 'at least'."
            raise ValueError(msg)

        self.relation = relation
        self.num_words = max(0, num_words)  # Ensure non-negative number of words.

    def check(self, response: str) -> bool:
        """Checks if the response meets the word count requirement.

        Args:
            response: A string representing the response.

        Returns:
            True if the response satisfies the word count constraint; otherwise, False.
        """
        num_words = count_words(response)

        if self.relation == "less than":
            return num_words < self.num_words
        if self.relation == "at least":
            return num_words >= self.num_words
        msg = f"Invalid relation: {self.relation}"
        raise ValueError(msg)


def count_words(text: str) -> int:
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    return len(tokens)
