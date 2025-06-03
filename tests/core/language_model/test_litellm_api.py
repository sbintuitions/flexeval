import os
from unittest.mock import patch

import pytest

from flexeval.core.language_model import LanguageModel, LiteLLMChatAPI
from flexeval.core.language_model.base import LMOutput
from flexeval.core.language_model.openai_api import OpenAIChatAPI

from .base import BaseLanguageModelTest


def is_openai_enabled() -> bool:
    return os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture(scope="module")
def chat_lm() -> LiteLLMChatAPI:
    return LiteLLMChatAPI(
        "gpt-4o-mini-2024-07-18",
        default_gen_kwargs={"temperature": 0.0},
    )


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI API Key is not set")
class TestLiteLLMChatAPI(BaseLanguageModelTest):
    @pytest.fixture()
    def lm(self) -> LanguageModel:
        return LiteLLMChatAPI(
            "gpt-4o-mini-2024-07-18",
            default_gen_kwargs={"temperature": 0.0},
            developer_message="You are text completion model. "
            "Please provide the text likely to continue after the user input. "
            "Do not provide the answer or any other information.",
        )

    @pytest.fixture()
    def chat_lm(self, chat_lm: LiteLLMChatAPI) -> LanguageModel:
        return chat_lm

    @pytest.fixture()
    def chat_lm_for_tool_calling(self, chat_lm: LiteLLMChatAPI) -> LiteLLMChatAPI:
        return chat_lm

    @pytest.mark.skip(reason="Even with temperature 0.0, the output is not deterministic via API.")
    def test_batch_complete_text_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        pass

    @pytest.mark.skip(reason="Even with temperature 0.0, the output is not deterministic via API.")
    def test_batch_chat_response_is_not_affected_by_batch(self, chat_lm: LanguageModel) -> None:
        pass


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_chat_log_probs_for_multi_tokens(chat_lm: LiteLLMChatAPI) -> None:
    prompt_list = [[{"role": "user", "content": "Output a number from 1 to 3."}] for _ in range(2)]
    response_list = [{"role": "assistant", "content": "1"}, {"role": "assistant", "content": "4"}]
    with pytest.raises(NotImplementedError):
        chat_lm.compute_chat_log_probs(prompt_list, response_list)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_if_ignore_seed() -> None:
    chat_lm = LiteLLMChatAPI("gpt-4o-mini-2024-07-18", ignore_seed=True)
    chat_messages = [{"role": "user", "content": "Hello"}]
    with patch.object(OpenAIChatAPI, "_batch_generate_chat_response", return_value=[LMOutput("Hello!")]) as mock_method:
        chat_lm.generate_chat_response(chat_messages, temperature=0.7, seed=42)
        # `seed` parameter should be removed
        mock_method.assert_called_once_with([chat_messages], None, temperature=0.7)

    text = "Hello, I'm"
    with patch.object(OpenAIChatAPI, "_batch_complete_text", return_value=[LMOutput("ChatGPT.")]) as mock_method:
        chat_lm.complete_text(text, stop_sequences=None, max_new_tokens=None, temperature=0.7, seed=42)
        # `seed` parameter should be removed
        mock_method.assert_called_once_with([text], None, None, temperature=0.7)


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_if_not_ignore_seed() -> None:
    chat_lm = LiteLLMChatAPI("gpt-4o-mini-2024-07-18")
    chat_messages = [{"role": "user", "content": "Hello"}]
    with patch.object(OpenAIChatAPI, "_batch_generate_chat_response", return_value=[LMOutput("Hello!")]) as mock_method:
        chat_lm.generate_chat_response(chat_messages, temperature=0.7, seed=42)
        mock_method.assert_called_once_with([chat_messages], None, temperature=0.7, seed=42)

    text = "Hello, I'm"
    with patch.object(OpenAIChatAPI, "_batch_complete_text", return_value=[LMOutput("ChatGPT.")]) as mock_method:
        chat_lm.complete_text(text, stop_sequences=None, max_new_tokens=None, temperature=0.7, seed=42)
        mock_method.assert_called_once_with([text], None, None, temperature=0.7, seed=42)
