from __future__ import annotations

from flexeval.core.utils.jinja2_utils import JINJA2_ENV


def test_regex_replace() -> None:
    template = "{{ text | regex_replace('<<.*?>>', '') }}"
    assert JINJA2_ENV.from_string(template).render(text="<<a>>Hello <<dummy>>world!") == "Hello world!"


def test_truncate_middle() -> None:
    s = "This is a pen"  # len(s) = 13
    assert truncate_middle(s, length=14, filler="+") == "This is a pen"
    assert truncate_middle(s, length=13, filler="+") == "This i+ a pen"
    assert truncate_middle(s, length=12, filler="+") == "This + a pen"
