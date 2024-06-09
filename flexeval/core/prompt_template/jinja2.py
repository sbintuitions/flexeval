from __future__ import annotations

from typing import Any

from flexeval.core.utils.jinja2_env import JINJA2_ENV

from .base import PromptTemplate


class Jinja2PromptTemplate(PromptTemplate):
    """
    Embed task inputs using Jinja2 template engine.

    Args:
        template: The Jinja2 template to use.
    """

    def __init__(self, template: str) -> None:
        self._template = template

    def embed_input(self, input_dict: dict[str, Any]) -> str:
        return JINJA2_ENV.from_string(self._template).render(input_dict)
