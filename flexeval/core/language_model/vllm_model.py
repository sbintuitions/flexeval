from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer

from .base import LanguageModel
from .hf_lm import normalize_stop_sequences


class VLLM(LanguageModel):
    """
    LanguageModel implementation using VLLM.

    Args:
        model: The name of the model to use.
        model_kwargs: Additional keyword arguments to pass to the model.
        tokenizer: The name of the tokenizer to use. Defaults to the model_name.
        tokenizer_kwargs: Keyword arguments for the tokenizer instantiation by `from_pretrained().
        add_special_tokens: Whether to add special tokens to the input.
            Note that whether BOS or EOS tokens are added depends on the tokenizer.
        custom_chat_template: A custom chat template for chatbot models.
            If specified, this overrides the default chat template of the tokenizer.
    """

    def __init__(
        self,
        model: str,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        add_special_tokens: bool = False,
        custom_chat_template: str | None = None,
    ) -> None:
        self.model_name = model
        tokenizer = tokenizer if tokenizer else model
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)
        self.custom_chat_template = custom_chat_template
        self.add_special_tokens = add_special_tokens

        from vllm import LLM

        model_kwargs = model_kwargs or {}
        self.llm = LLM(model, trust_remote_code=True, **model_kwargs)

    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        kwargs = kwargs.copy()  # avoid modifying the original kwargs

        # use greedy decoding by default
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0

        # absorb the stop_sequences and max_new_tokens into the kwargs
        if max_new_tokens is not None:
            if "max_tokens" in kwargs:
                msg = (
                    "`max_new_tokens` will be normalized to `max_tokens` before fed into the VLLM module."
                    "You can not specify both."
                )
                raise ValueError(msg)
            kwargs["max_tokens"] = max_new_tokens

        stop_sequences = normalize_stop_sequences(
            stop_sequences=stop_sequences,
            stop_from_kwargs=kwargs.pop("stop", None),
            eos_token=self.tokenizer.eos_token,
            ignore_eos=kwargs.get("ignore_eos", False),
        )

        model_inputs = self.tokenizer(
            text_list,
            add_special_tokens=self.add_special_tokens,
            return_token_type_ids=False,
        )

        from vllm import SamplingParams

        vllm_outputs = self.llm.generate(
            prompt_token_ids=model_inputs.input_ids,
            sampling_params=SamplingParams(**kwargs, stop=stop_sequences),
            use_tqdm=False,
        )
        generated_texts = [self.tokenizer.decode(outputs.outputs[0].token_ids) for outputs in vllm_outputs]

        # The `include_stop_str_in_output` option does not work, because we let llm generate tokens, not strings.
        # We manually remove the stop sequences from the generated texts.
        if not kwargs.get("include_stop_str_in_output", False):
            for stop in stop_sequences:
                for i, gen_text in enumerate(generated_texts):
                    stop_index = gen_text.find(stop)
                    if stop_index != -1:
                        generated_texts[i] = gen_text[:stop_index]
        return generated_texts

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str]:
        chat_messages_as_string = [
            self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.custom_chat_template,
            )
            for chat_messages in chat_messages_list
        ]
        return self.batch_complete_text(chat_messages_as_string, **kwargs)

    def __repr__(self) -> str:
        return f"VLLM(model_name={self.model_name})"
