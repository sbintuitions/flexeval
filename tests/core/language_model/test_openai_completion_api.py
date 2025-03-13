import os

import pytest

from flexeval.core.language_model import OpenAICompletionAPI
from flexeval.core.language_model.base import LanguageModel

from .base import BaseLanguageModelTest


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def lm() -> OpenAICompletionAPI:
    return OpenAICompletionAPI(model="gpt-3.5-turbo-instruct")


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
class TestOpenAICompletionAPI(BaseLanguageModelTest):
    @pytest.fixture()
    def lm(self, lm: OpenAICompletionAPI) -> LanguageModel:
        return lm

    @pytest.fixture()
    def chat_lm(self, lm: OpenAICompletionAPI) -> LanguageModel:
        return lm
