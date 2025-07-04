import os

import pytest

from flexeval.core.language_model import OpenAICompletionAPI
from flexeval.core.language_model.base import LanguageModel

from .base import BaseLanguageModelTest


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def lm() -> OpenAICompletionAPI:
    return OpenAICompletionAPI(
        model="gpt-3.5-turbo-instruct",
        default_gen_kwargs={"max_new_tokens": 16, "stop_sequences": "\n", "temperature": 0.0},
    )


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
class TestOpenAICompletionAPI(BaseLanguageModelTest):
    @pytest.fixture()
    def lm(self, lm: OpenAICompletionAPI) -> LanguageModel:
        return lm

    @pytest.fixture()
    def chat_lm(self, lm: OpenAICompletionAPI) -> LanguageModel:
        return lm

    @pytest.fixture()
    def chat_lm_for_tool_calling(self, lm: OpenAICompletionAPI) -> LanguageModel:
        return lm

    @pytest.mark.skip(reason="Even with temperature 0.0, the output is not deterministic via API.")
    def test_batch_complete_text_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        pass

    @pytest.mark.skip(reason="Even with temperature 0.0, the output is not deterministic via API.")
    def test_batch_chat_response_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        pass
