from __future__ import annotations

import tiktoken

from .base import Tokenizer


class TiktokenTokenizer(Tokenizer):
    def __init__(self, tokenizer_name: str | None = None, model_name: str | None = None) -> None:
        # raise error, if both tokenizer_name and model_name are provided
        if tokenizer_name is not None and model_name is not None:
            msg = "Only one of tokenizer_name or model_name must be provided."
            raise ValueError(msg)

        if tokenizer_name:
            self.encoding = tiktoken.get_encoding(tokenizer_name)
        elif model_name:
            self.encoding = tiktoken.encoding_for_model(model_name)
        else:
            msg = "Either tokenizer_name or model_name must be provided"
            raise ValueError(msg)

    def tokenize(self, text: str) -> list[str]:
        token_ids = self.encoding.encode(text)
        return [self.encoding.decode([token_id]) for token_id in token_ids]
