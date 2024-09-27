from __future__ import annotations

import re

import jinja2
from jinja2 import sandbox


def regex_replace(s: str, pattern: str, replace: str) -> str:
    return re.sub(pattern, replace, s)


JINJA2_ENV = sandbox.ImmutableSandboxedEnvironment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,  # important for not escaping double quotations
    keep_trailing_newline=True,
)
JINJA2_ENV.filters["regex_replace"] = regex_replace
