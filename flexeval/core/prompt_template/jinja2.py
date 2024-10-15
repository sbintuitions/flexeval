from __future__ import annotations

from typing import Any

from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import PromptTemplate


class Jinja2PromptTemplate(PromptTemplate):
    """
    Embed task inputs using Jinja2 template engine.

    Args:
        template: The Jinja2 template to use.
        template_path: The path to a file with the Jinja2 template to use.
    """

    def __init__(self, template: str | None = None, template_path: str | None = None) -> None:
        if template is None and template_path is None:
            msg = "Either template or template_path must be provided"
            raise ValueError(msg)
        if template is not None and template_path is not None:
            msg = "Only one of template or template_path can be provided"
            raise ValueError(msg)

        if template_path is not None:
            with open(template_path) as f:
                self.template = f.read()
        else:
            self.template = template

    def embed_inputs(self, input_dict: dict[str, Any]) -> str:
        return JINJA2_ENV.from_string(self.template).render(input_dict)

    def __repr__(self) -> str:
        return f"Jinja2PromptTemplate(template={self.template!r})"
