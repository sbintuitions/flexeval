import pytest

from flexeval.core.string_processor import TemplateRenderer


@pytest.mark.parametrize(
    ("template", "text", "expected"),
    [
        ("This is a {{text}}.", "cat", "This is a cat."),
        ("${{text}}$", "10", "$10$"),
        ("\\boxed{%raw%}{{%endraw%}{{text}}{%raw%}}{%endraw%}", "10.5", "\\boxed{10.5}"),
    ],
)
def test_template_renderer(template: str, text: str, expected: str) -> None:
    renderer = TemplateRenderer(template)
    assert renderer(text) == expected
