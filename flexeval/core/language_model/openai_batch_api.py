from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from enum import Enum
from pprint import pformat
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from openai.types import Batch

from .base import LanguageModel, LMOutput, normalize_stop_sequences
from .openai_api import number_of_tokens_in_openai_model, remove_duplicates_from_prompt_list

MAX_NUM_TRIALS = 3


class Status(str, Enum):
    # From: https://platform.openai.com/docs/guides/batch/getting-started
    validating = "validating"
    failed = "failed"
    in_progress = "in_progress"
    finalizing = "finalizing"
    completed = "completed"
    expired = "expired"
    cancelling = "cancelling"
    canceled = "canceled"


def create_request_details(model: str, custom_id: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": messages, **kwargs},
    }


class OpenAIChatBatchAPI(LanguageModel):
    """LanguageModel implementation using OpenAI's ChatGPT API for Batch API.
    NOTE: Batch size should be more than or equal to the size of the given dataset for efficient generation.

    Args:
        model: The name of the model to use.
        api_headers: A dictionary of headers to use when making requests to the OpenAI API.
        polling_interval_seconds: The interval in seconds to poll the batch status.
        default_gen_kwargs: Default generation kwargs to use when calling the API.
    """

    def __init__(
        self,
        model: str,
        api_headers: dict[str, str] | None = None,
        polling_interval_seconds: int = 60,
        default_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        if api_headers is None:
            api_headers = {}
        self._client = AsyncOpenAI(**api_headers)
        self.default_gen_kwargs = default_gen_kwargs or {}
        # convert the flexeval-specific argument name to the OpenAI-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_completion_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")
        self.temp_jsonl_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")

        self.polling_interval_seconds = polling_interval_seconds

    def create_batch_file(self, custom_id_2_message: dict[str, list[dict[str, str]]], **kwargs) -> None:
        with open(self.temp_jsonl_file.name, mode="w") as f:
            for custom_id, message in custom_id_2_message.items():
                f.write(
                    json.dumps(create_request_details(self.model, custom_id, message, **kwargs), ensure_ascii=False)
                    + "\n",
                )

    async def _post_batch_requests(
        self,
        custom_id_2_message: dict[str, list[dict[str, str]]],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> str:
        gen_kwargs = self.default_gen_kwargs.copy()
        gen_kwargs.update(kwargs)
        """Send batch chat requests to the OpenAI."""

        if max_new_tokens is not None:
            if "max_completion_tokens" in gen_kwargs:
                msg = (
                    "You specified both `max_new_tokens` and `max_completion_tokens` in generation kwargs. "
                    "Note that `max_new_tokens` overrides `max_completion_tokens` by default. "
                    "It is recommended to specify only one of them to avoid unexpected behavior."
                )
                logger.warning(msg)
            gen_kwargs["max_completion_tokens"] = max_new_tokens

        gen_kwargs["stop"] = normalize_stop_sequences(
            stop_sequences_list=[
                stop_sequences,
                gen_kwargs.pop("stop", None),  # This is used in the OpenAI API
                gen_kwargs.pop("stop_sequences", None),  # This is a common variable name used in flexeval
            ],
        )

        self.create_batch_file(custom_id_2_message, **gen_kwargs)

        # Update batch file
        with open(self.temp_jsonl_file.name, "rb") as batch_file:  # noqa: ASYNC101
            batch_input_file = await self._client.files.create(file=batch_file, purpose="batch")

        # Run Job
        # Batch Object: https://platform.openai.com/docs/api-reference/batch/object
        batch_object = await self._client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "flexeval job"},
        )
        logger.info(f"Input File ID: {batch_input_file.id}, Batch ID: {batch_object.id}")
        return batch_object.id

    async def poll_batch_status_until_completion(
        self,
        batch_id: str,
        polling_interval_seconds: int,
    ) -> tuple[Status, Batch]:
        status = Status.validating
        while status not in (Status.completed, Status.failed, Status.canceled):
            await asyncio.sleep(polling_interval_seconds)
            batch_response = await self._client.batches.retrieve(batch_id)
            status = Status(batch_response.status)
            logger.info(f"Current status: {status.value}")
        return status, batch_response

    def _retrieve_file_content(self, file_id: str) -> list[dict[any, any]]:
        file_response = asyncio.run(self._client.files.content(file_id))
        return [json.loads(line) for line in file_response.text.strip().split("\n")]

    def _execute_batch_requests(
        self,
        messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[Any]:
        custom_id_2_message: dict[str, list[dict[str, str]]] = {
            str(uuid.uuid4()): messages for messages in messages_list
        }
        # The response will be an empty string if the API produces an error.
        custom_id_2_response: dict[str, str | list[dict[str, str]]] = {
            custom_id: "" for custom_id in custom_id_2_message
        }
        exec_cnt = 1

        while len(custom_id_2_message) > 0:
            if exec_cnt > MAX_NUM_TRIALS:
                break
            logger.info(f"Trial {exec_cnt}")
            exec_cnt += 1
            batch_id = asyncio.run(self._post_batch_requests(custom_id_2_message, **kwargs))

            status, batch_response = asyncio.run(
                self.poll_batch_status_until_completion(batch_id, self.polling_interval_seconds),
            )
            if status is not Status.completed:
                error_message = f"Failed: {batch_response}"
                raise ValueError(error_message)

            # Check error_file_id exists and if exists, log error details.
            error_file_id = batch_response.error_file_id
            # If any request fails, error_file_id is set.
            if error_file_id is not None:
                logger.warning("Request on some messages failed following reason.")
                data: list[dict[str, Any]] = self._retrieve_file_content(error_file_id)
                # [Error](https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8857])
                # instance is embedded in response.
                for data_i in data:
                    error = data_i["response"]
                    logger.warning(f"Failed: {error}")

            output_file_id = batch_response.output_file_id
            # If completion on all input fails, output_file_id is None.
            if output_file_id is None:
                logger.warning("All request failed. Continue...")
                continue

            data: list[dict[str, Any]] = self._retrieve_file_content(output_file_id)
            for data_i in data:
                if data_i["error"] is not None:
                    continue

                custom_id = data_i["custom_id"]
                custom_id_2_message.pop(custom_id)

                custom_id_2_response[custom_id] = data_i["response"]["body"]

        # The remaining elements are all those that failed to complete request.
        if custom_id_2_message:
            logger.warning("The following messages failed to complete request.")
            logger.warning(pformat(list(custom_id_2_message.values())))

        return list(custom_id_2_response.values())

    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        messages_list = [[{"role": "user", "content": text}] for text in text_list]
        api_responses = self._execute_batch_requests(
            messages_list,
            stop_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return [
            LMOutput(text=res["choices"][0]["message"]["content"], finish_reason=res["choices"][0]["finish_reason"])
            for res in api_responses
        ]

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[LMOutput]:
        api_responses = self._execute_batch_requests(
            chat_messages_list,
            **kwargs,
        )
        return [
            LMOutput(text=res["choices"][0]["message"]["content"], finish_reason=res["choices"][0]["finish_reason"])
            for res in api_responses
        ]

    def close(self) -> None:
        # in case that the program fails before the file is initialized in __init__
        if not hasattr(self, "temp_jsonl_file"):
            return

        try:
            self.temp_jsonl_file.close()
            os.unlink(self.temp_jsonl_file.name)  # noqa: PTH108
            logger.info(f"Temporary file deleted: {self.temp_jsonl_file.name}")
        except OSError as e:
            logger.error(f"Error: {e.filename} - {e.strerror}.")

    def batch_compute_chat_log_probs(
        self,
        prompt_list: list[list[dict[str, str]]],
        response_list: list[dict[str, str]],
        temperature: float = 0,
        seed: int = 42,
        top_logprobs: int = 20,
    ) -> list[float]:
        response_contents = [resp["content"] for resp in response_list]
        for response_content in response_contents:
            num_tokens = number_of_tokens_in_openai_model(self.model, response_content)
            if num_tokens > 1:
                err_msg = f"OpenAIChatAPI.batch_compute_chat_log_probs is not applicable for two or more tokens of response content: '{response_content}'"  # noqa: E501
                raise NotImplementedError(err_msg)

        # For saving cost, remove duplication from message_list for an API request.
        unique_prompt_list = remove_duplicates_from_prompt_list(prompt_list)
        api_responses = self._execute_batch_requests(
            unique_prompt_list,
            max_new_tokens=1,
            seed=seed,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        log_probs = []
        top_logprobs_list = [res["choices"][0]["logprobs"]["content"][0]["top_logprobs"] for res in api_responses]
        for index, prompt in enumerate(prompt_list):
            target_token = response_contents[index]
            index_in_unique = unique_prompt_list.index(prompt)

            log_prob = None  # if target token not in top_logprobs, return None for log_prob of the token
            top_logprobs = top_logprobs_list[index_in_unique]
            for token_logprob in top_logprobs:
                if token_logprob["token"] == target_token:
                    log_prob = token_logprob["logprob"]
                    break
            log_probs.append(log_prob)

        return log_probs

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
