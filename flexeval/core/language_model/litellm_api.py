from __future__ import annotations

import os
from typing import Any, TypeVar

from litellm import ModelResponse, completion
from litellm.utils import convert_to_model_response_object

from flexeval.core.language_model.base import LMOutput
from flexeval.core.string_processor import StringProcessor

from .openai_api import EMPTY_RESPONSE as EMPTY_RESPONSE_OPENAI
from .openai_api import OpenAIChatAPI

T = TypeVar("T")

# LiteLLM uses `AZURE_API_BASE` as the environment variable for the AzureOpenAI endpoint,
# whereas the OpenAI SDK uses `AZURE_OPENAI_ENDPOINT`.
# For convenience, if only `AZURE_OPENAI_ENDPOINT` is set,
# also make it available to LiteLLM by assigning it to `AZURE_API_BASE`.
if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_API_BASE") is None:
    os.environ["AZURE_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]


class LiteLLMChatAPI(OpenAIChatAPI):
    """
    LanguageModel implementation using LiteLLM.
    Various APIs are available, such as OpenAI, Claude, Gemini, etc.
    See also: https://docs.litellm.ai/docs/providers

    Args:
        model: The name of the model to use. e.g. 'openai/gpt-3.5-turbo',
        default_gen_kwargs: Default generation kwargs to use when calling the API.
        developer_message: Instructions to the model that are prioritized ahead of user messages.
            Previously called the system prompt.
        string_processors: A single or a list of StringProcessor objects to process the model's output.
        ignore_seed: If True, ignore the seed specified in default_gen_kwargs.
            This is an option for models that do not support seed parameters such as anthropic/claude.
        model_limit_completion_tokens: An upper limit on the number of tokens the model can generate.
            For example, if a too-large `max_new_tokens` is given to generate_chat_response(), this value will cap it.
        max_parallel_requests: Maximum number of parallel requests to send to the API.
        tools: Default tools to use in chat responses when no tools are explicitly provided.
    """

    def __init__(
        self,
        model: str = "openai/gpt-3.5-turbo",
        default_gen_kwargs: dict[str, Any] | None = None,
        developer_message: str | None = None,
        string_processors: StringProcessor | list[StringProcessor] | None = None,
        ignore_seed: bool = False,
        model_limit_completion_tokens: int | None = None,
        max_parallel_requests: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_headers=None,
            default_gen_kwargs=default_gen_kwargs,
            developer_message=developer_message,
            string_processors=string_processors,
            model_limit_new_tokens=model_limit_completion_tokens,
            max_parallel_requests=max_parallel_requests,
            tools=tools,
            backend=None,
        )
        self.model = model
        self.default_gen_kwargs = default_gen_kwargs or {}
        # convert the flexeval-specific argument name to the OpenAI-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")

        self.empty_response = convert_to_model_response_object(
            response_object=EMPTY_RESPONSE_OPENAI.to_dict(),
            model_response_object=ModelResponse(),
        )
        self.ignore_seed = ignore_seed
        self.api_call_func = completion

    def set_random_seed(self, seed: int) -> None:
        self.default_gen_kwargs["seed"] = seed

    def _batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        if "seed" in kwargs and self.ignore_seed:
            kwargs.pop("seed")
        return super()._batch_complete_text(text_list, stop_sequences, max_new_tokens, **kwargs)

    def _batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, Any]]],
        tools_list: list[list[dict[str, Any]] | None] | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        if "seed" in kwargs and self.ignore_seed:
            kwargs.pop("seed")
        return super()._batch_generate_chat_response(chat_messages_list, tools_list, **kwargs)

    def _batch_compute_chat_log_probs(
        self,
        prompt_list: list[list[dict[str, Any]]],
        response_list: list[dict[str, Any]],
        temperature: float = 0,
        seed: int = 42,
        top_logprobs: int = 20,
    ) -> list[float | None]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
