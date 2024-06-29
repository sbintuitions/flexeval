from __future__ import annotations

from typing import Any

import pytest

from flexeval.core.utils.jinja2_utils import JINJA2_ENV, get_template_filter_function


def test_regex_replace() -> None:
    template = "{{ text | regex_replace('<<.*?>>', '') }}"
    assert JINJA2_ENV.from_string(template).render(text="<<a>>Hello <<dummy>>world!") == "Hello world!"


@pytest.mark.parametrize(
    ("template_str", "value_to_keep", "true_items", "false_items"),
    [
        (
            "{{ type }}",
            "filter_this",
            [
                {"type": "filter_this", "text": "Hello"},
                {"type": "filter_this", "text": "World"},
            ],
            [
                {"type": "not_filter_this", "text": "Hello"},
                {"type": "not_filter_this", "text": "World"},
            ],
        ),
        (
            "{{ answers | length }}",
            "1",
            [
                {"question": "Hello", "answers": ["World"]},
                {"question": "World", "answers": ["Hello"]},
            ],
            [
                {"question": "Hello", "answers": ["World", "Hello"]},
                {"question": "World", "answers": ["Hello", "World"]},
            ],
        ),
    ],
)
def test_get_template_filter_function(
    template_str: str,
    value_to_keep: str,
    true_items: list[dict[str, Any]],
    false_items: list[dict[str, Any]],
) -> None:
    template_filter_function = get_template_filter_function(template_str, value_to_keep)
    for item in true_items:
        assert template_filter_function(item)

    for item in false_items:
        assert not template_filter_function(item)
