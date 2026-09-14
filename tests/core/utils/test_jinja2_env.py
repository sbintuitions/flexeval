from __future__ import annotations

from flexeval.core.utils.jinja2_utils import JINJA2_ENV


def test_regex_replace() -> None:
    template = "{{ text | regex_replace('<<.*?>>', '') }}"
    assert JINJA2_ENV.from_string(template).render(text="<<a>>Hello <<dummy>>world!") == "Hello world!"


def test_truncate_middle() -> None:
    template = JINJA2_ENV.from_string("{{ text | truncate_middle(14, '+') }}")
    assert template.render(text="This is a pen") == "This is a pen"
    template = JINJA2_ENV.from_string("{{ text | truncate_middle(13, '+') }}")
    assert template.render(text="This is a pen") == "This i+ a pen"
    template = JINJA2_ENV.from_string("{{ text | truncate_middle(12, '+') }}")
    assert template.render(text="This is a pen") == "This + a pen"


def test_literal_eval() -> None:
    # parses a stringified list (a common storage format for options columns on the HF Hub)
    template = JINJA2_ENV.from_string(
        "{% for option in options | literal_eval %}{{ 'AB'[loop.index0] }}. {{ option }}\n{% endfor %}",
    )
    assert template.render(options="['cat', 'dog']") == "A. cat\nB. dog\n"
    # non-string values pass through unchanged
    assert template.render(options=["cat", "dog"]) == "A. cat\nB. dog\n"
