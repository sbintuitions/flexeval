from flexeval.core.utils.jinja2_utils import JINJA2_ENV

from .base import StringProcessor


class TemplateRenderer(StringProcessor):
    """Render a jinja2 template with a given string

    Examples:
        >>> from flexeval import TemplateRenderer
        >>> processor = TemplateRenderer("This is a {{text}}")
        >>> text = "ABC"
        >>> normalized_text = processor(text)
        >>> print(normalized_text)
        This is a ABC
    """

    def __init__(self, template: str) -> None:
        self._template = JINJA2_ENV.from_string(template)

    def __call__(self, text: str) -> str:
        return self._template.render(text=text)
