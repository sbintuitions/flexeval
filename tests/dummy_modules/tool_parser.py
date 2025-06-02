from flexeval.core.tool_parser import FunctionToolCall, ParsedToolCallingMessage, ToolParser


class DummyToolParser(ToolParser):
    def __call__(self, text: str) -> ParsedToolCallingMessage:
        return ParsedToolCallingMessage(
            "CompleteToolCall",
            text,
            f"{text}<|tool_calls|>[{{'name': 'get_weather', 'arguments': {{'city': 'Paris'}}}}]",
            [FunctionToolCall("get_weather", {"city": "Paris"}, "id1")],
        )
