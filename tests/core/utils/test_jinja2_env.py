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
