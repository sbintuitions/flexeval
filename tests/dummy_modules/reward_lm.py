from __future__ import annotations

from flexeval.core.language_model import LanguageModel


class DummyRewardLanguageModel(LanguageModel):
    """常に受け取ったresponseを出力する言語モデルです。 Reward Modelのテスト用に用意しています。"""
    def __init__(self, response: str = "[[A]]") -> None:
        super().__init__()
        self.response = response

    def batch_complete_text(self, text_list: list[str], **kwargs) -> list[str]:
        return [self.response for _ in text_list]

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
        return [self.response for _ in chat_messages_list]
