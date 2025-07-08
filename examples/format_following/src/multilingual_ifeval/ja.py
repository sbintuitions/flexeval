# Copyright 2004 The Google Research Authors.
# Copyright 2025 Lightblue
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

import collections
import json
import re
import unicodedata
import warnings

import langdetect

from examples.format_following.src.base import ResponseConstraint

from .ja_util import count_sentences, tokenizing_texts

# The relational operation for comparison.
_COMPARISON_RELATION = ("未満", "以上")

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = ("はい、そうです。", "いいえ、違います。", "どちらとも言えません。")


class ResponseLanguage(ResponseConstraint):
    """Check the language of the entire response."""

    def __init__(self, language: str) -> None:
        self.language = language

    def check(self, response: str) -> bool:
        try:
            return langdetect.detect(response) == self.language
        except langdetect.LangDetectException as e:
            warnings.warn(f"Unable to detect language for text '{response}' due to {e}", stacklevel=2)
            # If language detection fails, default to considering the instruction followed.
            return True


class NumberOfSentences(ResponseConstraint):
    """Check the number of sentences."""

    def __init__(self, num_sentences: int, relation: str) -> None:
        """
        Args:
            num_sentences: An integer specifying the number of sentences as a threshold.
            relation: A string in {'未満', '以上'}, defining the relational operator for comparison.
                - '未満': actual number of sentences < threshold
                - '以上': actual number of sentences >= threshold
        """
        self.num_sentences = num_sentences

        if relation not in _COMPARISON_RELATION:
            msg = f"The supported relation for comparison must be in {_COMPARISON_RELATION}, but '{relation}' is given."
            raise ValueError(
                msg,
            )
        self.relation = relation

    def check(self, response: str) -> bool:
        num = count_sentences(response)
        if self.relation == "未満":
            return num < self.num_sentences
        if self.relation == "以上":
            return num >= self.num_sentences
        msg = "Invalid relation for comparison."
        raise ValueError(msg)


class PlaceholderConstraint(ResponseConstraint):
    """Check the placeholders in template writing."""

    def __init__(self, num_placeholders: int = 1) -> None:
        """
        Args:
            num_placeholders: An integer denoting the minimum number of placeholders required in the response.
        """
        if num_placeholders < 0:
            msg = "num_placeholders must be non-negative"
            raise ValueError(msg)

        self.num_placeholders = num_placeholders

    def check(self, response: str) -> bool:
        placeholders = re.findall(r"\[.*?\]", response)
        return len(placeholders) >= self.num_placeholders


class BulletListChecker(ResponseConstraint):
    """Checks the bullet list in the prompt."""

    def __init__(self, num_bullets: int = 3) -> None:
        """
        Args:
            num_bullets: An integer specifying the exact number of bullet lists
                required to appear in the response.
        """
        if num_bullets < 0:
            msg = "num_bullets must be non-negative"
            raise ValueError(msg)
        self.num_bullets = num_bullets

    def check(self, response: str) -> bool:
        bullet_lists = re.findall(r"^\s*・[^\・].*$", response, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists)
        return num_bullet_lists == self.num_bullets


class NumberedList(ResponseConstraint):
    """Checks the numbered list in the response."""

    def __init__(self, num_items: int = 3) -> None:
        """
        Args:
            num_items: An integer specifying the exact number of numbered lists
                required to appear in the response.
        """
        if num_items < 0:
            msg = "num_items must be non-negative"
            raise ValueError(msg)

        self.num_items = num_items

    def check(self, response: str) -> bool:
        numbered_lists = re.findall(r"^\s*\d+\.\s.*$", response, flags=re.MULTILINE)
        num_numbered_lists = len(numbered_lists)
        return num_numbered_lists == self.num_items


class ConstrainedResponse(ResponseConstraint):
    """Checks the constrained response."""

    def __init__(self, constrained_responses: list[str] | None = None) -> None:
        """
        Args:
            constrained_responses: A list of strings representing the valid response options.
        """
        self.constrained_responses = constrained_responses or _CONSTRAINED_RESPONSE_OPTIONS

    def check(self, response: str) -> bool:
        response = response.strip()
        return any(constrained_response in response for constrained_response in self.constrained_responses)


class ConstrainedStart(ResponseConstraint):
    """Checks the response start."""

    def __init__(self, starter: str) -> None:
        """
        Args:
            starter: A string representing the keyword that the response should start with.
        """
        self.starter = starter.strip()

    def check(self, response: str) -> bool:
        response_pattern = r"^\s*" + re.escape(self.starter) + r".*$"
        response_with_constrained_start = re.search(response_pattern, response, flags=re.MULTILINE)
        return bool(response_with_constrained_start)


class HighlightSection(ResponseConstraint):
    """Checks the highlighted section."""

    def __init__(self, num_highlights: int = 1) -> None:
        """
        Args:
            num_highlights: An integer specifying the minimum number of highlighted sections.
        """
        if num_highlights < 0:
            msg = "num_highlights must be non-negative"
            raise ValueError(msg)

        self.num_highlights = num_highlights

    def check(self, response: str) -> bool:
        num_highlights = 0
        highlights = re.findall(r"《[^\n《》]*》", response)
        for highlight in highlights:
            if highlight.strip("《》").strip():
                num_highlights += 1

        return num_highlights >= self.num_highlights


class SectionChecker(ResponseConstraint):
    """Checks the sections."""

    def __init__(self, section_spliter: str = "章", num_sections: int = 1) -> None:
        """
        Args:
            section_spliter: A string representing the keyword that marks a new section, like '章' or '節'.
            num_sections: An integer specifying the minimum number of sections.
        """
        if not section_spliter.strip():
            msg = "section_spliter cannot be empty or whitespace."
            raise ValueError(msg)
        if num_sections < 0:
            msg = "num_sections must be non-negative"
            raise ValueError(msg)

        self.section_spliter = section_spliter.strip()
        self.num_sections = num_sections

    def check(self, response: str) -> bool:
        section_spliter_pattern = r"\s?" + r"第[\d\uFF10-\uFF19]+" + self.section_spliter + r"\s?"
        sections = re.split(section_spliter_pattern, response)
        num_sections = len(sections) - 1
        return num_sections >= self.num_sections


class ParagraphChecker(ResponseConstraint):
    """Check the number of paragraphs in the response."""

    def __init__(self, num_paragraphs: int = 3) -> None:
        """
        Args:
            num_paragraphs: An integer specifying the expected number of paragraphs.
        """
        if num_paragraphs < 0:
            msg = "num_paragraphs must be non-negative"
            raise ValueError(msg)
        self.num_paragraphs = num_paragraphs

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


class PostscriptChecker(ResponseConstraint):
    """Checks the postscript."""

    def __init__(self, postscript_marker: str = "P.S.") -> None:
        """
        Args:
            postscript_marker: A string containing the keyword that marks the
                start of the postscript section.
        """
        self.postscript_marker = postscript_marker.strip() if isinstance(postscript_marker, str) else postscript_marker

    def check(self, response: str) -> bool:
        if self.postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self.postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + re.escape(self.postscript_marker) + r".*$"
        postscript = re.findall(postscript_pattern, response, flags=re.IGNORECASE | re.MULTILINE)
        return bool(postscript)


class RephraseChecker(ResponseConstraint):
    """Checks the rephrase."""

    def __init__(self, original_message: str) -> None:
        """
        Args:
            original_message: A string representing the original message.
                The rephrased response should only change its words/sentences in between
                its two curly brackets, e.g., {change me}.
        """
        self.original_placeholder_values = self.extract_change_places(original_message)
        if not self.original_placeholder_values:
            msg = f"Message '{original_message}' does not contain changes in the form of {{change me}}."
            raise ValueError(msg)
        self.reference_without_change = original_message

    def check(self, response: str) -> bool:
        placeholder_values = self.extract_change_places(response)

        if len(self.original_placeholder_values) != len(placeholder_values):
            return False

        if self.original_placeholder_values == placeholder_values:
            return False

        response_without_changes = self.strip_changes(response)
        reference_without_changes = self.strip_changes(self.reference_without_change)

        return response_without_changes == reference_without_changes

    @staticmethod
    def extract_change_places(response: str) -> list[str]:
        """Check if there is change in the response in the form of {change me}."""
        return re.findall(r"\{.*?\}", response)

    @staticmethod
    def strip_changes(response: str) -> str:
        """Strips off the changes."""
        return re.sub(r"\{.*\}", "", response)


class KeywordChecker(ResponseConstraint):
    """Check the existence of certain keywords."""

    def __init__(self, keywords: list[str]) -> None:
        """
        Args:
            keywords: A list of strings representing the keywords expected in the response.
        """
        self.keywords = sorted(keywords)

    def check(self, response: str) -> bool:
        return all(keyword in response for keyword in self.keywords)


class KeywordFrequency(ResponseConstraint):
    """Check the keyword frequency."""

    def __init__(self, keyword: str, frequency: int = 1, relation: str = "以上") -> None:
        """
        Args:
            keyword: A string representing a keyword expected in the response.
            frequency: An integer specifying how many times `keyword` should appear.
            relation: A string in {'未満', '以上'} to define comparison:
                - '未満': actual occurrences < frequency
                - '以上': actual occurrences >= frequency
        """
        if relation not in {"未満", "以上"}:
            msg = f"relation must be one of ['未満', '以上'], but got '{relation}'"
            raise ValueError(msg)
        if frequency < 0:
            msg = "frequency must be non-negative"
            raise ValueError(msg)

        self.keyword = keyword.strip()
        self.frequency = frequency
        self.relation = relation

    def check(self, response: str) -> bool:
        actual_occurrences = response.count(self.keyword)
        if self.relation == "未満":
            return actual_occurrences < self.frequency
        if self.relation == "以上":
            return actual_occurrences >= self.frequency
        msg = "Invalid relation for comparison."
        raise ValueError(msg)


class NumberOfLetters(ResponseConstraint):
    """Checks the number of letters in the response."""

    def __init__(self, num_letters: int = 100, relation: str = "以上") -> None:
        """
        Args:
            num_letters: An integer specifying the number of letters expected in the response.
            relation: A string in {'未満', '以上'} determining how to compare the letter count.
                - '未満': actual number of letters < num_letters
                - '以上': actual number of letters >= num_letters
        """
        if relation not in {"未満", "以上"}:
            msg = f"relation must be one of ['未満', '以上'], but got '{relation}'"
            raise ValueError(msg)
        if num_letters < 0:
            msg = "num_letters must be non-negative"
            raise ValueError(msg)

        self.num_letters = num_letters
        self.relation = relation

    def check(self, response: str) -> bool:
        num_letters = len(response)
        if self.relation == "未満":
            return num_letters < self.num_letters
        if self.relation == "以上":
            return num_letters >= self.num_letters
        msg = "Invalid relation for comparison."
        raise ValueError(msg)


class JsonFormat(ResponseConstraint):
    """Check the JSON format."""

    def check(self, response: str) -> bool:
        value = (
            response.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError as _:
            return False
        return True


class ParagraphFirstWordCheck(ResponseConstraint):
    """Check the paragraph count and the first word of the nth paragraph."""

    def __init__(self, num_paragraphs: int = 3, nth_paragraph: int = 1, first_word: str = "default") -> None:
        """
        Args:
            num_paragraphs: An integer indicating the expected number of paragraphs.
            nth_paragraph: An integer indicating which paragraph's first word to check (1-based index).
            first_word: A string representing the expected first word of the nth paragraph.
        """
        if num_paragraphs < 1:
            msg = "num_paragraphs must be at least 1"
            raise ValueError(msg)
        if nth_paragraph < 1 or nth_paragraph > num_paragraphs:
            msg = "nth_paragraph must be between 1 and num_paragraphs inclusive"
            raise ValueError(msg)

        self.num_paragraphs = num_paragraphs
        self.nth_paragraph = nth_paragraph
        self.first_word = first_word.lower()

    def check(self, response: str) -> bool:
        paragraphs = re.split(r"\n\n", response)
        num_paragraphs = len(paragraphs)

        # Remove empty paragraphs
        paragraphs = [p for p in paragraphs if p.strip()]
        num_paragraphs = len(paragraphs)

        if self.nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self.nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        # Remove potential Japanese quotation marks
        paragraph = paragraph.lstrip("「")
        paragraph = paragraph.lstrip("『")

        first_word = paragraph[: len(self.first_word)]

        return num_paragraphs == self.num_paragraphs and first_word == self.first_word


class KeySentenceChecker(ResponseConstraint):
    """Check the existence of certain key sentences."""

    def __init__(self, key_sentences: list[str], num_sentences: int = 1) -> None:
        """
        Args:
            key_sentences: A list of strings representing the key sentences
                that are expected in the response.
            num_sentences: The number of key sentences that are expected to be
                seen in the response.
        """
        self.key_sentences = key_sentences

        if num_sentences < 1:
            msg = "num_sentences must be at least 1"
            raise ValueError(msg)
        self.num_sentences = num_sentences

    def check(self, response: str) -> bool:
        for key_sentence in self.key_sentences:
            count = response.count(key_sentence)
            if count != self.num_sentences:
                return False
        return True


class ForbiddenWords(ResponseConstraint):
    """Checks that specified words are not used in response."""

    def __init__(self, forbidden_words: list[str]) -> None:
        """
        Args:
            forbidden_words: A list of strings representing words that are not allowed in the response.
        """
        self.forbidden_words = sorted(set(forbidden_words))

    def check(self, response: str) -> bool:
        return all(forbidden_word not in response for forbidden_word in self.forbidden_words)


class RephraseParagraph(ResponseConstraint):
    """Check that the paragraph is rephrased."""

    def __init__(self, original_paragraph: str, low: int, high: int) -> None:
        """
        Args:
            original_paragraph: A string representing the original paragraph. The
                rephrased response should have between `low` and `high` words in common.
            low: An integer representing the lower bound of similar words.
            high: An integer representing the upper bound of similar words.
        """
        self.original_paragraph = original_paragraph
        self.low = low
        self.high = high

    def check(self, response: str) -> bool:
        tokens_value = tokenizing_texts(response)
        tokens_original = tokenizing_texts(self.original_paragraph)

        val_words = [
            token.surface
            for token in tokens_value
            if not (
                token.part_of_speech.startswith("助詞")
                or token.part_of_speech.startswith("助動詞")
                or token.part_of_speech.startswith("記号")
            )
        ]

        original_words = [
            token.surface
            for token in tokens_original
            if not (
                token.part_of_speech.startswith("助詞")
                or token.part_of_speech.startswith("助動詞")
                or token.part_of_speech.startswith("記号")
            )
        ]

        common_words = set(val_words).intersection(set(original_words))
        return self.low <= len(common_words) <= self.high


class TwoResponses(ResponseConstraint):
    """Check that two responses were given."""

    def check(self, response: str) -> bool:
        responses = response.split("******")
        valid_responses = [resp.strip() for resp in responses if resp.strip()]

        # Verify that there are exactly two valid responses and they are different
        return len(valid_responses) == 2 and valid_responses[0] != valid_responses[1]


class RepeatPromptThenAnswer(ResponseConstraint):
    """Checks that the prompt is first repeated then answered."""

    def __init__(self, prompt_to_repeat: str) -> None:
        """
        Args:
            prompt_to_repeat: The prompt that is meant to be repeated.
        """
        if not prompt_to_repeat:
            msg = "prompt_to_repeat must be set."
            raise ValueError(msg)
        self.prompt_to_repeat = prompt_to_repeat

    def check(self, response: str) -> bool:
        return response.strip().startswith(self.prompt_to_repeat.strip())


class EndChecker(ResponseConstraint):
    """Checks that the response ends with a given phrase."""

    def __init__(self, end_phrase: str) -> None:
        """
        Args:
            end_phrase: A string representing the phrase the response should end with.
        """
        self.end_phrase = end_phrase.strip()

    def check(self, response: str) -> bool:
        response = response.strip().strip("」』")
        end_phrase = self.end_phrase.strip().strip("」』")
        return response.endswith(end_phrase)


class TitleChecker(ResponseConstraint):
    """Checks the response for a title."""

    def check(self, response: str) -> bool:
        pattern = r"『[^\n]+』"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, response)

        return any(title.lstrip("『").rstrip("』").strip() for title in titles)


class LetterFrequency(ResponseConstraint):
    """Check letter frequency in the response."""

    def __init__(self, letter: str, let_frequency: int = 1, let_relation: str = "以上") -> None:
        """
        Args:
            letter: A string representing a letter expected in the response.
            let_frequency: An integer specifying the expected number of appearances of the letter.
            let_relation: A string in {'未満', '以上'} defining the comparison operator.
                - '未満': actual occurrences < frequency
                - '以上': actual occurrences >= frequency
        """

        if len(letter) != 1:
            msg = "letter must be a single character"
            raise ValueError(msg)

        self.letter = letter.strip()

        if let_frequency < 0:
            msg = "frequency must be non-negative"
            raise ValueError(msg)
        self.frequency = let_frequency

        if let_relation not in {"未満", "以上"}:
            msg = f"relation must be one of ['未満', '以上'], but got '{let_relation}'"
            raise ValueError(msg)
        self.relation = let_relation

    def check(self, response: str) -> bool:
        letters = collections.Counter(response)
        katakana_letter = chr(ord(self.letter) + 96)
        total_count = letters[self.letter] + letters[katakana_letter]

        if self.relation == "未満":
            return total_count < self.frequency
        return total_count >= self.frequency


class PeriodChecker(ResponseConstraint):
    """Checks the response for no periods."""

    def check(self, response: str) -> bool:
        return not re.search(r"\。", response)


class CommaChecker(ResponseConstraint):
    """Checks the response for no commas."""

    def check(self, response: str) -> bool:
        return not re.search(r"\、", response)


class QuotationChecker(ResponseConstraint):
    """Checks if the response is wrapped with Japanese quotation marks."""

    def check(self, response: str) -> bool:
        response = response.strip()
        return len(response) > 1 and response[0] == "「" and response[-1] == "」"


class FuriganaForKanji(ResponseConstraint):
    """Checks that all kanji are accompanied by furigana."""

    def __init__(self) -> None:
        pass

    def check(self, response: str) -> bool:
        """Checks if all kanji are described with furigana."""
        kanji_pattern = r"[\u4e00-\u9faf]+"
        kanji_with_furigana_pattern = r"[\u4e00-\u9faf]+（[ぁ-ん]+）"

        kanji_count = len(re.findall(kanji_pattern, response))
        kanji_with_furigana_count = len(re.findall(kanji_with_furigana_pattern, response))

        return kanji_count == kanji_with_furigana_count


class KanjiLimit(ResponseConstraint):
    """Check the number of Kanji used in the response."""

    def __init__(self, kanji_limit: int = 10, relation: str = "以上") -> None:
        """
        Args:
            kanji_limit: An integer specifying the number of kanji to be used.
            relation: A string in {'未満', '以上'} defining the relational operator for comparison.
                - '未満': actual number of kanji < kanji_limit
                - '以上': actual number of kanji >= kanji_limit
        """
        if relation not in {"未満", "以上"}:
            msg = f"relation must be one of ['未満', '以上'], but got '{relation}'"
            raise ValueError(msg)
        if kanji_limit < 0:
            msg = "kanji_limit must be non-negative"
            raise ValueError(msg)

        self.kanji_limit = kanji_limit
        self.relation = relation

    def check(self, response: str) -> bool:
        kanji_count = len(re.findall(r"[\u4e00-\u9faf]", response))
        if self.relation == "未満":
            return kanji_count < self.kanji_limit
        if self.relation == "以上":
            return kanji_count >= self.kanji_limit
        msg = "Invalid relation for comparison."
        raise ValueError(msg)


class NoHiragana(ResponseConstraint):
    """Checks that no Hiragana characters are used."""

    def check(self, response: str) -> bool:
        return not any("ぁ" <= char <= "ゖ" for char in response)


class HiraganaOnly(ResponseConstraint):
    """Checks if the response is written in Hiragana."""

    def __init__(self) -> None:
        pass

    def check(self, response: str) -> bool:
        def is_hiragana(char: str) -> str:
            return "ぁ" <= char <= "ん" or char == "ー"

        def is_ignorable(char: str) -> bool:
            return not unicodedata.category(char).startswith("L")

        return all(is_hiragana(char) or is_ignorable(char) for char in response)


class NoKatakana(ResponseConstraint):
    """Checks no Katakana."""

    def __init__(self) -> None:
        pass

    def check(self, response: str) -> bool:
        return not any(("ァ" <= char <= "ヺ" or "ｦ" <= char <= "ﾟ") for char in response)


class KatakanaOnly(ResponseConstraint):
    """Checks the response written in Katakana."""

    def check(self, response: str) -> bool:
        def is_katakana(char: str) -> bool:
            return "ァ" <= char <= "ン" or char == "ー" or char == "・" or "ｦ" <= char <= "ﾟ"

        def is_ignorable(char: str) -> bool:
            return not unicodedata.category(char).startswith("L")

        return all(is_katakana(char) or is_ignorable(char) for char in response)


class SentenceEndingUnification(ResponseConstraint):
    """Check all the sentence endings."""

    def __init__(self, ending: str = "です") -> None:
        """
        Args:
            ending: A string used at the end of all sentences.
        """
        self.ending = ending

    def check(self, response: str) -> bool:
        # Remove quoted text to focus on sentence endings outside quotes
        quote_pattern_1 = re.compile(r"「.*?」")
        quote_pattern_2 = re.compile(r"『.*?』")
        value = re.sub(quote_pattern_1, "", response)
        value = re.sub(quote_pattern_2, "", value)

        # Split sentences and check each ending
        sentences = re.split(r"[。！？]", value)
        return all(not (sentence and not sentence.endswith(self.ending)) for sentence in sentences)


class NominalEnding(ResponseConstraint):
    """Check the nominal endings in the response."""

    def __init__(self, count: int = 1) -> None:
        """
        Args:
            count: An integer specifying the exact number of nominal endings
                   required in the response.
        """
        if count < 1:
            msg = "count must be non-negative"
            raise ValueError(msg)

        self.count = count

    def check(self, response: str) -> bool:
        tokens = list(tokenizing_texts(response))

        noun_count = 0
        for i in range(1, len(tokens)):
            if tokens[i].surface in "。！？...」』\n" and tokens[i - 1].part_of_speech.startswith("名詞"):
                noun_count += 1
        noun_count += int(tokens[-1].part_of_speech.startswith("名詞"))
        return noun_count >= self.count


class KanjiNumberNotation(ResponseConstraint):
    """Ensure all numbers are written in kanji."""

    def check(self, response: str) -> bool:
        # Check if any digit is present in the response
        return not re.search(r"\d", response)
