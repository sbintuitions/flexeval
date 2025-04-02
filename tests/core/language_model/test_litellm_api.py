import os

import pytest

from flexeval.core.language_model import LanguageModel, LiteLLMChatAPI

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


@pytest.mark.skipif(not is_openai_enabled(), reason="OpenAI is not installed")
def test_compute_chat_log_probs_for_multi_tokens(chat_lm: LiteLLMChatAPI) -> None:
    prompt_list = [[{"role": "user", "content": "Output a number from 1 to 3."}] for _ in range(2)]
    response_list = [{"role": "assistant", "content": "1"}, {"role": "assistant", "content": "4"}]
    with pytest.raises(NotImplementedError):
        chat_lm.compute_chat_log_probs(prompt_list, response_list)


def test_ignore_seed() -> LiteLLMChatAPI:
    chat_lm = LiteLLMChatAPI(
        "gpt-4o-mini-2024-07-18",
        default_gen_kwargs={"temperature": 0.0, "seed": 42},
        ignore_seed=True
    )
    assert "seed" not in chat_lm.default_gen_kwargs
