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

import warnings

import langdetect

from examples.format_following.src.base import ResponseConstraint


class ResponseLanguage(ResponseConstraint):
    """Check the language of the entire response."""

    def __init__(self, language: str = "en") -> None:
        """
        Args:
            language: A string representing the expected language of the response.
                The language should comply with ISO 639-1 codes
                (e.g., 'en' for English, 'zh' for Chinese, 'fr' for French).
        """
        self.language = language

    def check(self, response: str) -> bool:
        try:
            return langdetect.detect(response) == self.language
        except langdetect.LangDetectException as e:
            warnings.warn(f"Unable to detect language for text '{response}' due to {e}", stacklevel=2)
            # If language detection fails, default to considering the instruction followed.
            return True
