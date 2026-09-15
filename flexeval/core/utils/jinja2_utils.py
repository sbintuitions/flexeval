from __future__ import annotations

import re
from ast import literal_eval as _ast_literal_eval
from typing import Any

import jinja2
from jinja2 import sandbox


def regex_replace(s: str, pattern: str, replace: str) -> str:
    return re.sub(pattern, replace, s)


def truncate_middle(s: str, length: int, filler: str = "...") -> str:
    if len(s) + len(filler) <= length:
        return s
    extract_len = length - len(filler)
    prefix_len = extract_len // 2
    suffix_len = extract_len // 2 + extract_len % 2
    return s[:prefix_len] + filler + s[-suffix_len:]


def literal_eval(value: Any) -> Any:  # noqa: ANN401
    """Parse a stringified Python literal (e.g. "['a', 'b']", a common storage
    format for list columns in Hugging Face datasets) into the actual object.
    Non-string values are returned unchanged.
    """
    if isinstance(value, str):
        return _ast_literal_eval(value)
    return value


JINJA2_ENV = sandbox.ImmutableSandboxedEnvironment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,  # important for not escaping double quotations
    keep_trailing_newline=True,
)
JINJA2_ENV.filters["regex_replace"] = regex_replace
JINJA2_ENV.filters["truncate_middle"] = truncate_middle
JINJA2_ENV.filters["literal_eval"] = literal_eval
