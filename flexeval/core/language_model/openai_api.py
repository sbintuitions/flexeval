from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

import openai
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from .base import LanguageModel, normalize_stop_sequences

T = TypeVar("T")


# NOTE: current implementation uses only choices[0].message.content field.
EMPTY_RESPONSE = ChatCompletion(
    id="dummy",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                content="", refusal=None, role="assistant", function_call=None, tool_calls=None
            ),
        )
    ],
    created=946652400,  # dummy integer
    model="dummy_model",
    object="chat.completion",
    service_tier=None,
    system_fingerprint=None,
    usage=None,
)


async def _retry_on_error(
    openai_call: Callable[[], Awaitable[T]],
    max_num_trials: int = 5,
    first_wait_time: int = 10,
) -> Awaitable[T]:
    for i in range(max_num_trials):
        try:
            return await openai_call()
        except openai.APIError as e:  # noqa: PERF203
            if i == max_num_trials - 1:
                # Since reaching maximum number of trials, exit for-loop and return
                # empty response.
                break
            logger.warning(f"We got an error: {e}")
            wait_time_seconds = first_wait_time * (2**i)
            logger.warning(f"Wait for {wait_time_seconds} seconds...")
            await asyncio.sleep(wait_time_seconds)

    logger.warning(f"We reached maximum number of trials ({max_num_trials} trials.).")
    logger.warning("Response including empty string is returned.")
    return EMPTY_RESPONSE


class OpenAIChatAPI(LanguageModel):
    """
    LanguageModel implementation using OpenAI's ChatGPT API.

    Args:
        model: The name of the model to use.
        api_headers: A dictionary of headers to use when making requests to the OpenAI API.
        default_gen_kwargs: Default generation kwargs to use when calling the API.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_headers: dict[str, str] | None = None,
        default_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        if api_headers is None:
            api_headers = {}
        self._client = AsyncOpenAI(**api_headers)
        self.default_gen_kwargs = default_gen_kwargs or {}
        # convert the flexeval-specific argument name to the OpenAI-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")

    async def _async_batch_run_chatgpt(
        self,
        messages_list: list[list[dict[str, str]]],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Send multiple chat requests to the OpenAI in parallel."""

        gen_kwargs = self.default_gen_kwargs.copy()
        gen_kwargs.update(kwargs)
        if max_new_tokens is not None:
            gen_kwargs["max_tokens"] = max_new_tokens

        stop_sequences = normalize_stop_sequences(
            stop_sequences_list=[
                stop_sequences,
                gen_kwargs.pop("stop", None),  # This is used in the OpenAI API
                gen_kwargs.pop("stop_sequences", None),  # This is a common variable name used in flexeval
            ],
        )

        tasks = [
            _retry_on_error(
                # Define an anonymous function with a lambda expression and pass it,
                # and call it inside the _retry_on_error function
                openai_call=lambda x=ms: self._client.chat.completions.create(
                    model=self.model,
                    messages=x,
                    stop=stop_sequences,
                    **gen_kwargs,
                ),
            )
            for ms in messages_list
        ]
        return await asyncio.gather(*tasks)

    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        messages_list = [[{"role": "user", "content": text}] for text in text_list]
        api_responses = asyncio.run(
            self._async_batch_run_chatgpt(
                messages_list,
                stop_sequences=stop_sequences,
                max_new_tokens=max_new_tokens,
                **kwargs,
            ),
        )
        completions = [res.choices[0].message.content for res in api_responses]
        if all(completion == "" for completion in completions):
            logger.warning("All generated texts are empty strings. Something may be wrong.")
        return completions

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str]:
        api_responses = asyncio.run(
            self._async_batch_run_chatgpt(chat_messages_list, **kwargs),
        )
        completions = [res.choices[0].message.content for res in api_responses]
        if all(completion == "" for completion in completions):
            logger.warning("All generated texts are empty string. Something may go wrong.")
        return completions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


class OpenAICompletionAPI(LanguageModel):
    """LanguageModel implementation using OpenAI's Completion API.

    Note that Completion API is a legacy API, with only a few models (such as gpt-3.5-turbo-instruct)
    supported by OpenAI. This LanguageModel implementation is primarily intended for use with on-premise
    VLLM servers, as described in the documentation: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

    Args:
        model: The name of the model to use.
        api_headers: A dictionary of headers to use when making requests to the OpenAI API.
        default_gen_kwargs: Default generation kwargs to use when calling the API.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct",
        api_headers: dict[str, str] | None = None,
        default_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        if api_headers is None:
            api_headers = {}
        self._client = AsyncOpenAI(**api_headers)
        self.default_gen_kwargs = default_gen_kwargs or {}
        # convert the flexeval-specific argument name to the OpenAI-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")

    async def _async_batch_run_completion(
        self,
        prompt_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Send multiple completion requests to the OpenAI in parallel."""

        gen_kwargs = self.default_gen_kwargs.copy()
        gen_kwargs.update(kwargs)
        if max_new_tokens is not None:
            gen_kwargs["max_tokens"] = max_new_tokens

        stop_sequences = normalize_stop_sequences(
            stop_sequences_list=[
                stop_sequences,
                gen_kwargs.pop("stop", None),  # This is used in the OpenAI API
                gen_kwargs.pop("stop_sequences", None),  # This is a common variable name used in flexeval
            ],
        )

        tasks = [
            _retry_on_error(
                # Define an anonymous function with a lambda expression and pass it,
                # and call it inside the _retry_on_error function
                openai_call=lambda x=ms: self._client.completions.create(
                    model=self.model,
                    prompt=x,
                    stop=stop_sequences,
                    **gen_kwargs,
                ),
            )
            for ms in prompt_list
        ]
        return await asyncio.gather(*tasks)

    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        api_responses = asyncio.run(
            self._async_batch_run_completion(
                text_list,
                stop_sequences=stop_sequences,
                max_new_tokens=max_new_tokens,
                **kwargs,
            ),
        )

        return [res.choices[0].text for res in api_responses]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
