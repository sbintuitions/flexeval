from __future__ import annotations

import re

import jinja2


def regex_replace(s: str, pattern: str, replace: str) -> str:
    return re.sub(pattern, replace, s)


JINJA2_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,  # noqa: S701
    keep_trailing_newline=True,
)
JINJA2_ENV.filters["regex_replace"] = regex_replace
