from __future__ import annotations

import itertools
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, TypeVar

import openai
import tiktoken
from loguru import logger
from openai import AzureOpenAI, BaseModel, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from flexeval.core.language_model.base import LanguageModel, LMOutput, normalize_stop_sequences
from flexeval.core.string_processor import StringProcessor

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


def _retry_on_error(
    openai_call: Callable[[], T],
    empty_response: BaseModel,
    max_num_trials: int | None = None,
    first_wait_time: int | None = None,
    max_wait_time: int | None = None,
) -> T:
    max_num_trials = max_num_trials or 5
    first_wait_time = first_wait_time or 10
    max_wait_time = max_wait_time or 80
    for i in range(max_num_trials):
        try:
            return openai_call()
        except openai.APIError as e:  # noqa: PERF203
            if i == max_num_trials - 1:
                # Since reaching maximum number of trials, exit for-loop and return
                # empty response.
                break
            logger.warning(f"We got an error: {e}")
            wait_time_seconds = min(max_wait_time, first_wait_time * (2**i))
            logger.warning(f"Wait for {wait_time_seconds} seconds...")
            time.sleep(wait_time_seconds)

    logger.warning(f"We reached maximum number of trials ({max_num_trials} trials.).")
    logger.warning("Response including empty string is returned.")
    return empty_response


class OpenAIChatAPI(LanguageModel):
    """
    LanguageModel implementation using OpenAI's ChatGPT API.
    Note that this class is inherited by litellm_api.LiteLLMChatAPI, so be careful when making any modifications.

    Args:
        model: The name of the model to use.
        api_headers: A dictionary of headers to use when making requests to the OpenAI API.
        default_gen_kwargs: Default generation kwargs to use when calling the API.
        developer_message: Instructions to the model that are prioritized ahead of user messages.
            Previously called the system prompt.
        string_processors: A single or a list of StringProcessor objects to process the model's output.
        model_limit_new_tokens: An upper limit on the number of tokens the model can generate.
            For example, if a too-large `max_new_tokens` is given to generate_chat_response(), this value will cap it.
        max_parallel_requests: Maximum number of parallel requests to send to the OpenAI API.
        tools: Default tools to use in chat responses when no tools are explicitly provided.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_headers: dict[str, str] | None = None,
        default_gen_kwargs: dict[str, Any] | None = None,
        developer_message: str | None = None,
        string_processors: StringProcessor | list[StringProcessor] | None = None,
        model_limit_new_tokens: int | None = None,
        max_parallel_requests: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_num_trials: int | None = None,
        first_wait_time: int | None = None,
        max_wait_time: int | None = None,
        backend: Literal["OpenAI", "AzureOpenAI"] | None = "OpenAI",
    ) -> None:
        super().__init__(string_processors=string_processors, tools=tools)
        self.model = model
        if api_headers is None:
            api_headers = {}
        self.empty_response = EMPTY_RESPONSE
        self.default_gen_kwargs = default_gen_kwargs or {}
        # convert the flexeval-specific argument name to the OpenAI-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_completion_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")

        self.developer_message = developer_message
        self.model_limit_new_tokens = model_limit_new_tokens
        self.max_parallel_requests = max_parallel_requests
        self.max_num_trials = max_num_trials
        self.first_wait_time = first_wait_time
        self.max_wait_time = max_wait_time
        if backend == "OpenAI":
            self.api_call_func = OpenAI(**api_headers).chat.completions.create
        elif backend == "AzureOpenAI":
            self.api_call_func = AzureOpenAI(**api_headers).chat.completions.create
        else:
            self.api_call_func = None

    def set_random_seed(self, seed: int) -> None:
        self.default_gen_kwargs["seed"] = seed

    def _parallel_run_chatgpt(
        self,
        messages_list: list[list[dict[str, Any]]],
        tools_list: list[list[dict[str, Any]] | None] | None = None,
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[ChatCompletion]:
        """Send multiple chat requests to the OpenAI in parallel."""
        if self.developer_message is not None:
            # Insert the developer message at the beginning of each conversation
            messages_list = [
                [{"role": "developer", "content": self.developer_message}, *messages] for messages in messages_list
            ]

        gen_kwargs = self.default_gen_kwargs.copy()
        gen_kwargs.update(kwargs)
        if max_new_tokens is not None:
            if "max_completion_tokens" in gen_kwargs:
                msg = (
                    "You specified both `max_new_tokens` and `max_completion_tokens` in generation kwargs. "
                    "Note that `max_new_tokens` overrides `max_completion_tokens` by default. "
                    "It is recommended to specify only one of them to avoid unexpected behavior."
                )
                logger.warning(msg)
            gen_kwargs["max_completion_tokens"] = max_new_tokens

        if self.model_limit_new_tokens and (gen_kwargs.get("max_completion_tokens", 0) > self.model_limit_new_tokens):
            msg = (
                f"The specified `max_new_tokens` ({gen_kwargs['max_completion_tokens']}) exceeds"
                f"the model’s capability ({self.model_limit_new_tokens} tokens). It will be reduced."
            )
            logger.warning(msg)
            gen_kwargs["max_completion_tokens"] = self.model_limit_new_tokens

        stop_sequences = normalize_stop_sequences(
            stop_sequences_list=[
                stop_sequences,
                gen_kwargs.pop("stop", None),  # This is used in the OpenAI API
                gen_kwargs.pop("stop_sequences", None),  # This is a common variable name used in flexeval
            ],
        )

        if tools_list is None:
            tools_list = [None] * len(messages_list)

        total = len(messages_list)
        # Progress logging controlled via the environment variable OPENAI_PROGRESS_EVERY_N
        # N>0: log every N completions; 0 or unset: disables
        prog_every_n = int(os.getenv("OPENAI_PROGRESS_EVERY_N", "0"))
        max_workers = self.max_parallel_requests or total
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, (messages, tools) in enumerate(zip(messages_list, tools_list)):
                future = executor.submit(
                    _retry_on_error,
                    openai_call=lambda messages=messages, tools=tools: self.api_call_func(
                        **{
                            **({"model": self.model, "messages": messages} | gen_kwargs),
                            **({"tools": tools} if tools else {}),
                            **({"stop": stop_sequences} if stop_sequences else {}),
                        }
                    ),
                    empty_response=self.empty_response,
                    max_num_trials=self.max_num_trials,
                    first_wait_time=self.first_wait_time,
                    max_wait_time=self.max_wait_time,
                )
                future_to_idx[future] = idx

            results: list[ChatCompletion] = [self.empty_response] * total
            for done_count, future in enumerate(as_completed(future_to_idx.keys()), start=1):
                idx = future_to_idx[future]
                results[idx] = future.result()
                if prog_every_n and (done_count % prog_every_n == 0 or done_count == total):
                    logger.info(f"[progress] {done_count}/{total} ({done_count / total:.1%}) done")

            return results

    def _batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        messages_list = [[{"role": "user", "content": text}] for text in text_list]
        api_responses = self._parallel_run_chatgpt(
            messages_list,
            stop_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        outputs = [
            LMOutput(
                text=res.choices[0].message.content,
                reasoning_text=getattr(res.choices[0].message, "reasoning_content", None),
                finish_reason=res.choices[0].finish_reason,
            )
            for res in api_responses
        ]

        if all(output.text == "" for output in outputs):
            logger.warning("All generated texts are empty strings. Something may be wrong.")
        return outputs

    def _batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, Any]]],
        tools_list: list[list[dict[str, Any]] | None] | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        api_responses = self._parallel_run_chatgpt(chat_messages_list, tools_list=tools_list, **kwargs)
        outputs = [
            LMOutput(
                text=res.choices[0].message.content,
                reasoning_text=getattr(res.choices[0].message, "reasoning_content", None),
                finish_reason=res.choices[0].finish_reason,
                tool_calls=[tool_call.to_dict() for tool_call in res.choices[0].message.tool_calls]
                if res.choices[0].message.tool_calls
                else None,
            )
            for res in api_responses
        ]
        if all(output.text == "" for output in outputs):
            logger.warning("All generated texts are empty strings. Something may go wrong.")
        return outputs

    def _batch_compute_chat_log_probs(
        self,
        prompt_list: list[list[dict[str, Any]]],
        response_list: list[dict[str, Any]],
        temperature: float = 0,
        seed: int = 42,
        top_logprobs: int = 20,
    ) -> list[float | None]:
        """
        Return logprob of one-token response only due to restriction of OpenAI API.
        If you pass a response with two or more tokens, raise an error.

        This function is mainly used for calculating weighted average of multi-choice prompts.
        Under the design of this function, we need to pass the same prompt (the number of choice) times.
        We only need one request for one prompt because OpenAI API returns a list of log probs.
        So, this function removes duplicates of prompts before requesting API and
        returns log probs by restoring the raw prompt list.
        """

        # Check the number of tokens is 1
        response_contents = [resp["content"] for resp in response_list]
        for response_content in response_contents:
            num_tokens = number_of_tokens_in_openai_model(self.model, response_content)
            if num_tokens > 1:
                err_msg = f"OpenAIChatAPI.batch_compute_chat_log_probs is not applicable for two or more tokens of response content: '{response_content}'"  # noqa: E501
                raise NotImplementedError(err_msg)

        # For saving cost, remove duplication from message_list for an API request.
        unique_prompt_list = remove_duplicates_from_prompt_list(prompt_list)
        api_responses = self._parallel_run_chatgpt(
            unique_prompt_list,
            max_completion_tokens=1,
            seed=seed,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        log_probs = []
        top_logprobs_list = [res.choices[0].logprobs.content[0].top_logprobs for res in api_responses]
        for index, prompt in enumerate(prompt_list):
            target_token = response_contents[index]
            index_in_unique = unique_prompt_list.index(prompt)

            log_prob = None  # if target token not in top_logprobs, return None for log_prob of the token
            top_logprobs = top_logprobs_list[index_in_unique]
            for token_logprob in top_logprobs:
                if token_logprob.token == target_token:
                    log_prob = token_logprob.logprob
                    break
            log_probs.append(log_prob)

        return log_probs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


def number_of_tokens_in_openai_model(model: str, content: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(content))


def message_list_from_prompt(prompt: list[dict[str, Any]]) -> list[str]:
    """A preprocess function to remove duplicates from prompt_list.
    This function translates prompt into list[str], allowing sorting
    """
    return [f"[{message['role']}]{message['content']}" for message in prompt]


def prompt_from_message_list(message_list: list[str]) -> list[dict[str, Any]]:
    """The inverted function of message_list_from_prompt."""
    prompt = []
    for message_str in message_list:
        role_end = message_str.index("]")
        role = message_str[1:role_end]
        content = message_str[role_end + 1 :]
        prompt.append({"role": role, "content": content})
    return prompt


def remove_duplicates_from_prompt_list(prompt_list: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """We cannot sort raw prompt_list because order is not defined for dict.

    Removing duplicates can be done as below.
    1. Change each element in prompt_list into list[str]
    2. Sort list of list[str] by itertools.groupby
    3. Invert list[str] into the original prompt format
    """
    messages_list = [message_list_from_prompt(prompt) for prompt in prompt_list]
    messages_list.sort()
    unique_messages_list = [element for element, _ in itertools.groupby(messages_list)]
    return [prompt_from_message_list(messages) for messages in unique_messages_list]


class OpenAICompletionAPI(LanguageModel):
    """LanguageModel implementation using OpenAI's Completion API.

    Note that Completion API is a legacy API, with only a few models (such as gpt-3.5-turbo-instruct)
    supported by OpenAI. This LanguageModel implementation is primarily intended for use with on-premise
    VLLM servers, as described in the documentation: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

    Args:
        model: The name of the model to use.
        api_headers: A dictionary of headers to use when making requests to the OpenAI API.
        default_gen_kwargs: Default generation kwargs to use when calling the API.
        string_processors: A single or a list of StringProcessor objects to process the model's output.
        max_parallel_requests: Maximum number of parallel requests to send to the OpenAI API.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct",
        api_headers: dict[str, str] | None = None,
        default_gen_kwargs: dict[str, Any] | None = None,
        string_processors: StringProcessor | list[StringProcessor] | None = None,
        max_parallel_requests: int | None = None,
    ) -> None:
        super().__init__(string_processors=string_processors)
        self.model = model
        if api_headers is None:
            api_headers = {}
        client = OpenAI(**api_headers)
        self.api_call_func = client.completions.create
        self.empty_response = EMPTY_RESPONSE
        self.default_gen_kwargs = default_gen_kwargs or {}
        self.max_parallel_requests = max_parallel_requests
        # convert the flexeval-specific argument name to the OpenAI-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")

    def _parallel_run_chatgpt(
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
        if stop_sequences:
            gen_kwargs["stop"] = stop_sequences

        max_workers = self.max_parallel_requests or len(prompt_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _retry_on_error,
                    # Define an anonymous function with a lambda expression and pass it,
                    # and call it inside the _retry_on_error function
                    openai_call=lambda x=ms: self.api_call_func(
                        model=self.model,
                        prompt=x,
                        **gen_kwargs,
                    ),
                    empty_response=self.empty_response,
                )
                for ms in prompt_list
            ]
        return [future.result() for future in futures]

    def set_random_seed(self, seed: int) -> None:
        self.default_gen_kwargs["seed"] = seed

    def _batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        api_responses = self._parallel_run_chatgpt(
            text_list,
            stop_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        return [LMOutput(text=res.choices[0].text, finish_reason=res.choices[0].finish_reason) for res in api_responses]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
