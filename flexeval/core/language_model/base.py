from __future__ import annotations


class LanguageModel:
    """LanguageModel is what you want to evaluate with this library.

    It can generate text based on the input text, response to chat messages, and compute log probabilities.
    """

    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate text based on the input text list.

        Args:
            text_list: A list of input texts.
            stop_sequences: A string or a list of strings that will stop the generation when they are generated.
            max_new_tokens: The maximum number of tokens to generate for each text.
        """
        msg = f"{self.__class__.__name__} cannot generate text."
        raise NotImplementedError(msg)

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str]:
        """Generate chat responses based on the chat messages in the list.
        This method is used for chatbot models.

        Args:
            chat_messages_list: A list of chat messages.
        """
        msg = f"{self.__class__.__name__} cannot generate chat responses."
        raise NotImplementedError(msg)

    def batch_compute_log_probs(
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
