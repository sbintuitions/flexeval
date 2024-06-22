from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PromptTemplate(ABC):
    """
    This class embeds task inputs from `GenerationInstance` or `MultipleChoiceInstance` into a text that can be used
    as a prompt for `LanguageModel`.
    """

    @abstractmethod
    def embed_inputs(self, input_dict: dict[str, Any]) -> str:
        """
        Embeds the input into a prompt template.
        """
        raise NotImplementedError
