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
        self.template = template

    def embed_inputs(self, input_dict: dict[str, Any]) -> str:
        return JINJA2_ENV.from_string(self.template).render(input_dict)

    def __repr__(self) -> str:
        return f"Jinja2PromptTemplate(template={self.template!r})"
