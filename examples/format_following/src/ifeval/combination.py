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

from examples.format_following.src.base import ResponseConstraint


class RepeatPrompt(ResponseConstraint):
    """
    Checks that Prompt is first repeated then answered.

    Args:
      prompt_to_repeat: The prompt that is meant to be repeated.
    """

    def __init__(self, prompt_to_repeat: str) -> None:
        self.prompt_to_repeat = prompt_to_repeat

    def check(self, response: str) -> bool:
        if response.strip().lower().startswith(self.prompt_to_repeat.strip().lower()):
            return True
        return False


class TwoResponses(ResponseConstraint):
    """
    Check that two responses were given.
    """

    def check(self, response: str) -> bool:
        valid_responses = [text.strip() for text in response.split("******") if text.strip()]

        return len(valid_responses) == 2 and valid_responses[0] != valid_responses[1]
