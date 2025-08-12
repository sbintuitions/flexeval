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

from examples.format_following.src.base import ResponseConstraint


class NumberPlaceholders(ResponseConstraint):
    """
    Check the placeholders in template writing.

    Args:
      num_placeholders: An integer denoting the minimum number of
        placeholders required in the response.
    """

    def __init__(self, num_placeholders: int) -> None:
        self.num_placeholders = num_placeholders

    def check(self, response: str) -> bool:
        placeholders = re.findall(r"\[.*?\]", response)
        num_placeholders = len(placeholders)
        return num_placeholders >= self.num_placeholders


class Postscript(ResponseConstraint):
    """
    Checks the postscript.

    Args:
      postscript_marker: A string containing the keyword that marks the start
        of the postscript section.
    """

    def __init__(self, postscript_marker: str) -> None:
        self.postscript_marker = postscript_marker

    def check(self, response: str) -> bool:
        value = response.lower()
        if self.postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self.postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self.postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return bool(postscript)
