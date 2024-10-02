from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from .base import LanguageModel, normalize_stop_sequences


def tokenize_text_for_lm_prefix(
    text_list: list[str],
    tokenizer: PreTrainedTokenizer,
    add_special_tokens: bool = False,
) -> list[list[int]]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer(
        text_list,
        return_tensors=None,
        padding=False,
        add_special_tokens=add_special_tokens,
        return_token_type_ids=False,
    )
    return model_inputs.input_ids


def tokenize_text_for_lm_continuation(
    text_list: list[str],
    tokenizer: PreTrainedTokenizer,
    oov_character: str = "彁",
    as_continuation: bool | list[bool] = True,
) -> list[list[int]]:
    if isinstance(as_continuation, bool):
        as_continuation = [as_continuation] * len(text_list)

    if len(as_continuation) != len(text_list):
        msg = "The length of as_continuation must be the same as the length of text_list."
        raise ValueError(msg)

    if oov_character in tokenizer.get_vocab():
        msg = f"oov_character '{oov_character}' is already in the tokenizer's vocab."
        raise ValueError(msg)
    oov_char_len = len(tokenizer.tokenize(oov_character))

    encoding_list: list[list[int]] = []
    for text, as_cont in zip(text_list, as_continuation):
        input_text = text
        # tokenize with OOV character
        if as_cont:
            input_text = oov_character + text
        encoding = tokenizer(
            input_text,
            add_special_tokens=False,
            return_token_type_ids=False,
        )
        # remove OOV character
        if as_cont:
            for k in encoding:
                encoding[k] = encoding[k][oov_char_len:]
        encoding_list.append(encoding.input_ids)

    return encoding_list


class VLLM(LanguageModel):
    """LanguageModel implementation using VLLM.

    Args:
        model: The name of the model to use.
        model_kwargs: Additional keyword arguments to pass to the model.
        tokenizer: The name of the tokenizer to use. Defaults to the model_name.
        tokenizer_kwargs: Keyword arguments for the tokenizer instantiation by `from_pretrained().
        add_special_tokens: Whether to add special tokens to the input.
            Note that whether BOS or EOS tokens are added depends on the tokenizer.
        custom_chat_template: A custom chat template for chatbot models.
            If specified, this overrides the default chat template of the tokenizer.
        default_gen_kwargs: Default generation kwargs to use when calling the model.
    """

    def __init__(
        self,
        model: str,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        add_special_tokens: bool = False,
        custom_chat_template: str | None = None,
        default_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model
        tokenizer = tokenizer if tokenizer else model
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)
        self.custom_chat_template = custom_chat_template
        self.add_special_tokens = add_special_tokens
        # use greedy decoding by default to make it consistent with `HuggingFaceLM`
        self.default_gen_kwargs = default_gen_kwargs or {"temperature": 0.0}
        # convert the flexeval-specific argument name to the vllm-specific name
        if "max_new_tokens" in self.default_gen_kwargs:
            self.default_gen_kwargs["max_tokens"] = self.default_gen_kwargs.pop("max_new_tokens")

        # import from vllm here because it is an extra dependency
        from vllm import LLM

        model_kwargs = model_kwargs or {}
        # automatically set tensor_parallel_size to the number of GPUs
        if "tensor_parallel_size" not in model_kwargs:
            model_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        if "enable_chunked_prefill" not in model_kwargs:
            model_kwargs["enable_chunked_prefill"] = True
            model_kwargs["disable_sliding_window"] = True
        self.llm = LLM(model, **model_kwargs)

    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        gen_kwargs = self.default_gen_kwargs.copy()
        gen_kwargs.update(kwargs)
        if max_new_tokens is not None:
            gen_kwargs["max_tokens"] = max_new_tokens

        stop_sequences = normalize_stop_sequences(
            stop_sequences_list=[
                stop_sequences,
                gen_kwargs.pop("stop", None),  # This is used in the vllm `SamplingParams`
                gen_kwargs.pop("stop_sequences", None),  # This is a common variable name used in flexeval
            ],
            eos_token=self.tokenizer.eos_token,
            ignore_eos=gen_kwargs.get("ignore_eos", False),
        )

        model_inputs = self.tokenizer(
            text_list,
            add_special_tokens=self.add_special_tokens,
            return_token_type_ids=False,
        )

        from vllm import SamplingParams

        vllm_outputs = self.llm.generate(
            prompt_token_ids=model_inputs.input_ids,
            sampling_params=SamplingParams(**gen_kwargs, stop=stop_sequences),
            use_tqdm=False,
        )
        generated_texts = [self.tokenizer.decode(outputs.outputs[0].token_ids) for outputs in vllm_outputs]

        # The `include_stop_str_in_output` option does not work, because we let llm generate tokens, not strings.
        # We manually remove the stop sequences from the generated texts.
        if not gen_kwargs.get("include_stop_str_in_output", False):
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

    def batch_compute_log_probs(
        self, text_list: list[str], prefix_list: list[str] | None = None, stride: int | None = None
    ) -> list[float]:
        batch_size = len(text_list)

        # prepare prefix encoding
        prefix_list = prefix_list if prefix_list else ["" for _ in range(batch_size)]
        # If the prefix is an empty string, replace it with the bos token regardless of the model being trained with it.
        # This is needed to correctly calculate the log probabilities of the first token.
        for i in range(batch_size):
            if prefix_list[i] == "":
                prefix_list[i] = self.tokenizer.bos_token

        batch_prefix_ids = tokenize_text_for_lm_prefix(
            prefix_list,
            self.tokenizer,
            add_special_tokens=self.add_special_tokens,
        )

        # prepare continuation encoding
        # If the last token is a special token, it is treated as a beginning of a new sentence.
        batch_continuation_ids = tokenize_text_for_lm_continuation(
            text_list,
            self.tokenizer,
            as_continuation=[prefix_ids[-1] not in self.tokenizer.all_special_ids for prefix_ids in batch_prefix_ids],
        )

        batch_input_ids = [
            prefix + continuation for prefix, continuation in zip(batch_prefix_ids, batch_continuation_ids)
        ]

        max_length = self.llm.llm_engine.get_model_config().max_seq_len_to_capture
        stride = stride or max_length // 2
        if not (0 < stride < max_length):
            msg = f"stride must be in (0, {max_length}), but got {stride}"
            raise ValueError(msg)
        sequence_length = max([len(input_ids) for input_ids in batch_input_ids])

        from vllm import RequestOutput, SamplingParams
        from vllm.sequence import Logprob

        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

        batch_logprobs = [0.0] * batch_size
        last_computed_index = 0
        for chunk_start in range(0, sequence_length, stride):
            chunk_end = min(chunk_start + max_length, sequence_length)
            chunk_batch_input_ids = [input_ids[chunk_start:chunk_end] for input_ids in batch_input_ids]
            chunk_batch_input_ids = [
                [self.tokenizer.bos_token_id] if len(chunk_input_ids) == 0 else chunk_input_ids
                for chunk_input_ids in chunk_batch_input_ids
            ]
            chunk_batch_outputs: list[RequestOutput] = self.llm.generate(
                prompt_token_ids=chunk_batch_input_ids,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            i = 0
            for ids, output, prefix_ids in zip(chunk_batch_input_ids, chunk_batch_outputs, batch_prefix_ids):
                chunk_rest_prefix_length = max(len(prefix_ids) - last_computed_index, 0)
                chunk_continuation_start = last_computed_index - chunk_start + chunk_rest_prefix_length

                # `prompt_logprobs` has the same length as the input `ids`.
                # The i-th element contains the log probabilities of the i-th token in `ids`
                # and the highest-likelihood token at that position.
                # The 0-th element is always `None` because the log probability cannot be computed for it.
                prompt_logprobs: list[dict[int, Logprob] | None] = output.prompt_logprobs
                all_token_logprobs = [
                    cands[token_id].logprob if cands else 0.0 for cands, token_id in zip(prompt_logprobs, ids)
                ]
                continuation_logprob = float(sum(all_token_logprobs[chunk_continuation_start:]))
                batch_logprobs[i] += continuation_logprob
                i += 1

            last_computed_index = chunk_end

        return batch_logprobs

    def __repr__(self) -> str:
        return f"VLLM(model_name={self.model_name})"
