import logging

from flexeval.core.language_model.base import LMOutput


def test_lmoutput_text_only() -> None:
    output = LMOutput(text="hello world")
    assert output.text == "hello world"
    assert output.raw_text is None
    assert output.finish_reason is None
    assert output.tool_calls is None
    assert output.tool_call_validation_result is None


def test_lmoutput_tool_calls_with_text() -> None:
    text = "This is a text."
    tool_calls = [{"name": "tool1", "args": {}}]
    output = LMOutput(text=text, tool_calls=tool_calls)
    assert output.text == text
    assert output.tool_calls == tool_calls


def test_lmoutput_tool_calls_without_text() -> None:
    tool_calls = [{"name": "tool1", "args": {}}]
    output = LMOutput(text=None, tool_calls=tool_calls)
    assert output.text == ""
    assert output.tool_calls == tool_calls


def test_lmoutput_empty_text_and_tool_calls(caplog) -> None:  # noqa: ANN001
    caplog.set_level(logging.WARNING)
    output = LMOutput(text=None, tool_calls=None)
    assert output.text == ""
    assert output.tool_calls is None
    assert len(caplog.records) >= 1
    assert any(record.msg.startswith("Both `text` and `tool_calls` are empty.") for record in caplog.records)
