import os

import pytest

from flexeval.core.language_model import OpenAICompletionAPI
from flexeval.core.language_model.base import LanguageModel, LMOutput

from .base import BaseLanguageModelTest


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def lm() -> OpenAICompletionAPI:
    return OpenAICompletionAPI(model="gpt-3.5-turbo-instruct")


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
class TestOpenAICompletionAPI(BaseLanguageModelTest):
    @pytest.fixture()
    def model(self, lm: OpenAICompletionAPI) -> LanguageModel:
        return lm

    @pytest.fixture()
    def chat_model(self, lm: OpenAICompletionAPI) -> LanguageModel:
        """OpenAICompletionAPI doesn't support chat, but we need to implement this fixture.
        Tests that require chat functionality will be skipped due to NotImplementedError.
        """
        return lm


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
def test_batch_generate_chat_response(lm: OpenAICompletionAPI) -> None:
    responses = lm.batch_complete_text(
        ["質問：Completion APIってlegacyなAPIなの？"],
        max_new_tokens=20,
        stop_sequences=["。"],
    )

    assert len(responses) == 1
    assert isinstance(responses[0], LMOutput)
    assert isinstance(responses[0].text, str)
    assert isinstance(responses[0].finish_reason, str)
