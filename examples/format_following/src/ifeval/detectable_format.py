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

import json
import re

from examples.format_following.src.base import ResponseConstraint


class ConstrainedResponse(ResponseConstraint):
    """Checks the constrained response."""

    def __init__(self, response_options: list[str] | None = None) -> None:
        """
        Args:
            response_options: A list of strings representing the possible response options.
        """
        self.response_options = response_options or ["My answer is yes.", "My answer is no.", "My answer is maybe."]

    def check(self, response: str) -> bool:
        response = response.strip()
        return any(option in response for option in self.response_options)


class JsonFormat(ResponseConstraint):
    """Check the Json format."""

    def check(self, response: str) -> bool:
        response = (
            response.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(response)
        except ValueError as _:
            return False
        return True


class MultipleSections(ResponseConstraint):
    """Checks the sections."""

    def __init__(self, section_spliter: str = "Section", num_sections: int = 1) -> None:
        """
        Args:
            section_spliter: A string that represents the section splitter keyword,
                e.g., "Section" or "SECTION".
            num_sections: An integer specifying the required number of sections.
        """
        self.section_spliter = section_spliter.strip()
        self.num_sections = num_sections if num_sections > 0 else 1

    def check(self, response: str) -> bool:
        """Checks if the response contains the required number of sections.

        Args:
            response: A string representing the response.

        Returns:
            True if the response contains the required number of sections;
            otherwise, False.
        """
        section_splitter_pattern = rf"\s?{re.escape(self.section_spliter)}\s?\d+\s?"
        sections = re.split(section_splitter_pattern, response)
        num_detected_sections = len(sections) - 1  # Subtract 1 because the first split part is not a section.
        return num_detected_sections >= self.num_sections


class NumberBulletLists(ResponseConstraint):
    """Checks the bullet list in the response."""

    def __init__(self, num_bullets: int = 1) -> None:
        """
        Args:
            num_bullets: An integer specifying the exact number of bullet points
                required in the response. Defaults to 1.
        """
        self.num_bullets = max(1, num_bullets)  # Ensure at least 1 bullet point.

    def check(self, response: str) -> bool:
        bullet_points_asterisks = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
        bullet_points_dashes = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
        total_bullet_points = len(bullet_points_asterisks) + len(bullet_points_dashes)
        return total_bullet_points == self.num_bullets


class NumberHighlightedSections(ResponseConstraint):
    """Checks the highlighted sections in the response."""

    def __init__(self, num_highlights: int = 1) -> None:
        """
        Args:
            num_highlights: An integer specifying the minimum number of highlighted
                sections required in the response. Defaults to 1.
        """
        self.num_highlights = max(1, num_highlights)  # Ensure at least 1 highlight.

    def check(self, response: str) -> bool:
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", response)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", response)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1
        return num_highlights >= self.num_highlights


class Title(ResponseConstraint):
    """Checks the response for a title."""

    def check(self, response: str) -> bool:
        titles = re.findall(r"<<[^\n]+>>", response)

        return any(title.lstrip("<").rstrip(">").strip() for title in titles)
