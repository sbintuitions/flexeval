from flexeval.core.tool_parser import FunctionToolCall, ToolCallingMessage, ToolParser


class DummyToolParser(ToolParser):
    def __call__(self, text: str) -> ToolCallingMessage:
        return ToolCallingMessage(
            "CompleteToolCall",
            text,
            f"{text}<|tool_calls|>[{{'name': 'get_weather', 'arguments': {{'city': 'Paris'}}}}]",
            [FunctionToolCall("get_weather", {"city": "Paris"}, "id1")],
        )
