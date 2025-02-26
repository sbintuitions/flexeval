from __future__ import annotations

from typing import Any, TypeVar

from litellm import ModelResponse, acompletion
from litellm.utils import convert_to_model_response_object

from .openai_api import EMPTY_RESPONSE as EMPTY_RESPONSE_OPENAI
from .openai_api import OpenAIChatAPI

T = TypeVar("T")


class LiteLLMChatAPI(OpenAIChatAPI):
    """
    LanguageModel implementation using LiteLLM.
    Various APIs are available, such as OpenAI, Claude, Gemini, etc.
    See also: https://docs.litellm.ai/docs/providers

    Args:
        model: The name of the model to use. e.g. 'openai/gpt-3.5-turbo',
        default_gen_kwargs: Default generation kwargs to use when calling the API.
    """

    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        default_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model, api_headers=None, default_gen_kwargs=default_gen_kwargs)
        self.model = model
        self.default_gen_kwargs = default_gen_kwargs or {}
        # convert the flexeval-specific argument name to the OpenAI-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")

        self.api_call_func = acompletion
        self.empty_response = convert_to_model_response_object(
            response_object=EMPTY_RESPONSE_OPENAI.to_dict(),
            model_response_object=ModelResponse(),
        )

    def batch_compute_chat_log_probs(
        self,
        prompt_list: list[list[dict[str, str]]],
        response_list: list[dict[str, str]],
        temperature: float = 0,
        seed: int = 42,
        top_logprobs: int = 20,
    ) -> list[float | None]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
