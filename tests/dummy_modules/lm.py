from __future__ import annotations

import json
from typing import Any

from flexeval.core.language_model import LanguageModel
from flexeval.core.language_model.base import LMOutput
from flexeval.core.tool_parser.base import FunctionToolCall


class DummyLanguageModel(LanguageModel):
    """入力をそのまま出力するダミーの言語モデルです。 テスト用に用意しています。"""

    def _batch_complete_text(self, text_list: list[str], **kwargs) -> list[LMOutput]:
        kwargs_as_text = json.dumps(kwargs)
        return [LMOutput(text=text + kwargs_as_text, finish_reason="length") for text in text_list]

    def _batch_compute_log_probs(
        self,
        text_list: list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> list[float]:
        return [-1.0] * len(text_list)

    def set_random_seed(self, seed: int) -> None:
        pass

    def _batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        tools_list: list[list[dict[str, Any]]] | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        tool_calls_list = [None for _ in chat_messages_list]
        if tools_list:
            for i, tools in enumerate(tools_list):
                if tools:
                    tool_calls = [
                        FunctionToolCall(tool["function"]["name"], '{"dummy_arg": "dummy_value"}').to_dict()
                        for tool in tools
                    ]
                    tool_calls_list[i] = tool_calls

        return [
            LMOutput(
                text=f"This is response to `{messages[-1]['content']}` with kwargs {kwargs}",
                reasoning_text="reasoning_text",
                finish_reason="length",
                tool_calls=tc,
                tool_call_validation_result="CompleteToolCall" if tc else "TextOnly",
            )
            for messages, tc in zip(chat_messages_list, tool_calls_list)
        ]
