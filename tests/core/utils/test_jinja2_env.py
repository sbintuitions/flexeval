from __future__ import annotations

from flexeval.core.utils.jinja2_utils import JINJA2_ENV


def test_regex_replace() -> None:
    template = "{{ text | regex_replace('<<.*?>>', '') }}"
    assert JINJA2_ENV.from_string(template).render(text="<<a>>Hello <<dummy>>world!") == "Hello world!"
