from __future__ import annotations

import functools
import re
from typing import Any, Callable

import jinja2


def regex_replace(s: str, pattern: str, replace: str) -> str:
    return re.sub(pattern, replace, s)


JINJA2_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,  # noqa: S701
)
JINJA2_ENV.filters["regex_replace"] = regex_replace


def get_template_filter_function(template_str: str, value_to_keep: str) -> Callable[[dict[str, Any]], bool]:
    """
    Get a filter function to filter items in a dataset.

    Args:
        template_str: The Jinja2 template string to embed the item into a string.
        value_to_keep: Keep the item if the rendered template string is equal to this value.

    Returns:
        A filter function to filter items in a dataset.
        This can be used as a filter function for `datasets.Dataset.filter`.
    """

    def _to_keep_this_item(item: dict[str, Any], filter_template: jinja2.Template, value_to_keep: str) -> bool:
        return filter_template.render(**item) == value_to_keep

    return functools.partial(
        _to_keep_this_item,
        filter_template=JINJA2_ENV.from_string(template_str),
        value_to_keep=value_to_keep,
    )
