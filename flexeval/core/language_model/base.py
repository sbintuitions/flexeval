from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final

from flexeval.core.string_processor import StringProcessor


@dataclass
class LMOutput:
    text: str | None
    """
    The output text of the language model.
    None is allowed only if tool_calls is set.
    """
    raw_text: str | None = None
    """
    The raw output text of the language model before post-processing.
    """
    finish_reason: str | None = None
    """
    The reason why the generation is finished.
    Typically,
    - 'stop': A stop sequence is generated.
    - 'length': The maximum length is reached.
    """
    tool_calls: list[dict[str, Any]] | None = None
    """
    the tools called by the language model
    """
    tool_call_validation_result: str | None = None
    """
    validation results of parsing for tool_calls
    """

    def __post_init__(self) -> None:
        if self.tool_calls is None and self.text is None:
            msg = "It is not allowed for both `text` and `tool_calls` to be None."
            raise ValueError(msg)


class LanguageModel:
    """LanguageModel is what you want to evaluate with this library.

    It can generate text based on the input text, response to chat messages, and compute log probabilities.

    Args:
        string_processors: A single or a list of StringProcessor objects to process the model's output.

    """

    def __init__(
        self,
        string_processors: StringProcessor | list[StringProcessor] | None = None,
    ) -> None:
        if string_processors is None:
            string_processors = []
        elif isinstance(string_processors, StringProcessor):
            string_processors = [string_processors]

        self.string_processors = string_processors

    def _batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        """
        Generate text based on the input text list.

        Args:
            text_list: A list of input texts.
            stop_sequences: A string or a list of strings that will stop the generation when they are generated.
                This argument exists to give a common interface to various models that have different names for it.
            max_new_tokens: The maximum number of tokens to generate for each text.
                This argument exists to give a common interface to various models that have different names for it.
            **kwargs: Additional keyword arguments for text generation.
                The acceptable keys depend on the specific implementation of the model.
                These arguments override corresponding values in the model's default_gen_kwargs.
                Special cases:
                - 'stop_sequences' or any similar model-specific kwargs:
                    Merged with default_gen_kwargs instead of overriding.
        """
        msg = f"{self.__class__.__name__} cannot generate text."
        raise NotImplementedError(msg)

    def _batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, Any]]],
        tools_list: list[list[dict[str, Any]]] | None = None,
        **kwargs,
    ) -> list[LMOutput]:
        """Generate chat responses based on the chat messages in the list.
        This method is used for chatbot models.

        Args:
            chat_messages_list: A list of chat messages.
            tools_list: A list of tool definitions.
                Each function definition should be a dict that conforms to the OpenAI Chat Completion API format.
                https://platform.openai.com/docs/guides/function-calling?api-mode=chat#defining-functions
        """
        msg = f"{self.__class__.__name__} cannot generate chat responses."
        raise NotImplementedError(msg)

    def _batch_compute_log_probs(
        self,
        text_list: list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> list[float]:
        """
        Compute log probabilities of the text list.
        Used for compute perplexity of text, or solving multiple choice questions.

        Args:
            text_list: A list of texts to compute log probabilities.
            prefix_list: A list of prefixes for each text.
            stride: The stride for computing log probabilities.
        """
        msg = f"{self.__class__.__name__} cannot compute perplexity."
        raise NotImplementedError(msg)

    def _batch_compute_chat_log_probs(
        self, prompt_list: list[list[dict[str, Any]]], response_list: list[dict[str, Any]]
    ) -> list[float]:
        """
        Compute log probabilities of the chat responses given the chat history.

        Args:
            prompt_list: A list of chat histories.
            response_list: A list of chat responses.
        """
        msg = f"{self.__class__.__name__} cannot compute chat log probabilities."
        raise NotImplementedError(msg)

    @final
    def complete_text(
        self,
        text: str | list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> LMOutput | list[LMOutput]:
        """
        A wrapper for `batch_complete_text` that accepts a single text or a list of texts.
        This is a convenient method for end-users.
        To implement generation logic, you should override `batch_complete_text` method.
        """

        # Normalize the input text
        text_list = text
        if isinstance(text, str):
            text_list = [text]

        lm_outputs = self._batch_complete_text(
            text_list, stop_sequences=stop_sequences, max_new_tokens=max_new_tokens, **kwargs
        )

        # Post-process the generated text
        if self.string_processors:
            for lm_output in lm_outputs:
                lm_output.raw_text = lm_output.text
                for string_processor in self.string_processors:
                    lm_output.text = string_processor(lm_output.text)

        # Return the result
        if isinstance(text, str):
            return lm_outputs[0]
        return lm_outputs

    @final
    def generate_chat_response(
        self,
        chat_messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        tools: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
        **kwargs,
    ) -> LMOutput | list[LMOutput]:
        """
        A wrapper for `batch_generate_chat_response` that accepts a single chat message or a list of chat messages.
        This is a convenient method for end-users.
        To implement generation logic, you should override `batch_generate_chat_response` method.
        """

        chat_messages_list = chat_messages
        tools_list = tools

        if isinstance(chat_messages[0], dict):
            chat_messages_list = [chat_messages]
        if tools and isinstance(tools[0], dict):
            tools_list = [tools]

        if tools_list and len(tools_list) != len(chat_messages_list):
            msg = "tools_list must be either None or a list of the same length as chat_messages_list."
            raise ValueError(msg)

        lm_outputs = self._batch_generate_chat_response(chat_messages_list, tools_list=tools_list, **kwargs)

        # Post-process the generatessd text
        if self.string_processors:
            for lm_output in lm_outputs:
                lm_output.raw_text = lm_output.text
                for string_processor in self.string_processors:
                    lm_output.text = string_processor(lm_output.text)

        # Return the result
        if isinstance(chat_messages[0], dict):
            return lm_outputs[0]
        return lm_outputs

    @final
    def compute_log_probs(
        self,
        text_list: str | list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> float | list[float]:
        """
        A wrapper for `batch_compute_log_probs` that accepts a single text or a list of texts.
        This is a convenient method for end-users.
        To implement computation logic, you should override `batch_compute_log_probs` method.
        """

        if isinstance(text_list, str):
            return self._batch_compute_log_probs([text_list], prefix_list=prefix_list, stride=stride)[0]
        return self._batch_compute_log_probs(text_list, prefix_list=prefix_list, stride=stride)

    @final
    def compute_chat_log_probs(
        self, prompt: list[dict[str, Any]] | list[list[dict[str, Any]]], response: dict[str, Any] | list[dict[str, Any]]
    ) -> float | list[float]:
        """
        A wrapper for `batch_compute_chat_log_probs` that accepts a single chat prompt or a list of chat prompts.
        This is a convenient method for end-users.
        To implement computation logic, you should override `batch_compute_chat_log_probs` method.
        """

        if isinstance(prompt[0], dict):
            return self._batch_compute_chat_log_probs([prompt], [response])[0]
        return self._batch_compute_chat_log_probs(prompt, response)

    def resource_cleanup(self) -> None:
        """
        Clean up resources if necessary.
        This method is called when the language model is no longer needed.
        """

    def __del__(self) -> None:
        self.resource_cleanup()

def normalize_stop_sequences(
    stop_sequences_list: list[str | list[str] | None],
    bos_token: str | None = None,
    eos_token: str | None = None,
    ignore_eos: bool = False,
) -> list[str]:
    """
    This function absorb stop sequences specified in various ways into a list of strings.
    """
    normalized_stop_sequences: list[str] = []
    # collect stop sequences from `stop_sequences_list`
    for stop_sequences in stop_sequences_list:
        if stop_sequences is None:
            pass
        elif isinstance(stop_sequences, str):
            normalized_stop_sequences.append(stop_sequences)
        elif isinstance(stop_sequences, list):
            normalized_stop_sequences.extend(stop_sequences)
        else:
            msg = f"Invalid type of stop_sequences: {type(stop_sequences)}"
            raise ValueError(msg)
    if bos_token:
        normalized_stop_sequences.append(bos_token)
    if eos_token and not ignore_eos:
        normalized_stop_sequences.append(eos_token)
    return list(set(normalized_stop_sequences))
