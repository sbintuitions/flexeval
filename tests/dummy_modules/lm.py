from __future__ import annotations

import json

from flexeval.core.language_model import LanguageModel
from flexeval.core.language_model.base import LMOutput


class DummyLanguageModel(LanguageModel):
    """入力をそのまま出力するダミーの言語モデルです。 テスト用に用意しています。"""

    def complete_text(self, text_list: list[str], **kwargs) -> list[LMOutput]:
        kwargs_as_text = json.dumps(kwargs)
        return [LMOutput(text=text + kwargs_as_text, finish_reason="length") for text in text_list]

    def compute_log_probs(
        self,
        text_list: list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> list[float]:
        return [-1.0] * len(text_list)

    def generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[LMOutput]:
        messages_as_text = [json.dumps(messages) for messages in chat_messages_list]
        kwargs_as_text = json.dumps(kwargs)
        return [LMOutput(text=m + kwargs_as_text, finish_reason="length") for m in messages_as_text]
