from __future__ import annotations

import json

from flexeval.core.language_model import LanguageModel


class DummyLanguageModel(LanguageModel):
    """入力をそのまま出力するダミーの言語モデルです。 テスト用に用意しています。"""

    def batch_complete_text(self, text_list: list[str], **kwargs) -> list[str]:
        kwargs_as_text = json.dumps(kwargs)
        return [text + kwargs_as_text for text in text_list]

    def batch_compute_log_probs(
        self,
        text_list: list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> list[float]:
        return [-1.0] * len(text_list)

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str]:
        return ["This is response."] * len(chat_messages_list)
