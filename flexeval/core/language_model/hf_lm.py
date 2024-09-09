from __future__ import annotations

import contextlib
from typing import Any, Literal, TypeVar

import torch
import torch.nn.functional as F  # noqa: N812
import transformers
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from .base import LanguageModel, normalize_stop_sequences

T = TypeVar("T")


@contextlib.contextmanager
def set_temporal_padding_side(tokenizer: PreTrainedTokenizer, padding_side: str) -> None:
    """Temporarily set the padding side of the tokenizer.

    Useful when the padding side of the tokenizer is unknown or needs to be overridden.
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original_padding_side


def tokenize_text_for_lm_prefix(
    text_list: list[str],
    tokenizer: PreTrainedTokenizer,
    add_special_tokens: bool = False,
) -> BatchEncoding:
    with set_temporal_padding_side(tokenizer, "left"):
        # use eos_token as pad_token if pad_token is None
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_inputs = tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
            return_token_type_ids=False,
        )

    # When an empty string is input, the dtype of the Tensor becomes float, so convert it to long.
    model_inputs.input_ids = model_inputs.input_ids.long()
    model_inputs.attention_mask = model_inputs.attention_mask.long()
    return model_inputs


def tokenize_text_for_lm_continuation(
    text_list: list[str],
    tokenizer: PreTrainedTokenizer,
    oov_character: str = "彁",
    as_continuation: bool | list[bool] = True,
) -> BatchEncoding:
    """When tokenizing a prefix and continuation separately, the sentencepiece-
    based tokenizer appends a special token to the beginning of the
    continuation.

    e.g.,
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    >>> tokenizer.tokenize("というのはテストです")
    ['▁', 'というのは', 'テスト', 'です']  # The first token is not necessary!

    To avoid this, we add a dummy character, which should never form a token with other characters, to the beginning
    of the continuation, and remove the dummy tokens after tokenization.

    If `as_continuation` is a boolean, it determines whether all texts should be treated as continuations.
    If `as_continuation` is a list of booleans, it specifies whether each text should be treated as a continuation.
    """
    if isinstance(as_continuation, bool):
        as_continuation = [as_continuation] * len(text_list)

    if len(as_continuation) != len(text_list):
        msg = "The length of as_continuation must be the same as the length of text_list."
        raise ValueError(msg)

    if oov_character in tokenizer.get_vocab():
        msg = f"oov_character '{oov_character}' is already in the tokenizer's vocab."
        raise ValueError(msg)
    oov_char_len = len(tokenizer.tokenize(oov_character))

    encoding_list: list[BatchEncoding] = []
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
        encoding_list.append(encoding)

    with set_temporal_padding_side(tokenizer, "right"):
        return tokenizer.pad(encoding_list, return_tensors="pt")


class HuggingFaceLM(LanguageModel):
    """
    LanguageModel implementation using Hugging Face Transformers.

    Args:
        model: The model name or path of the Hugging Face model.
        model_kwargs: Keyword arguments for the model instantiation by `from_pretrained()`.
        tokenizer: The tokenizer name or path of the Hugging Face tokenizer.
        tokenizer_kwargs: Keyword arguments for the tokenizer instantiation by `from_pretrained().
        add_special_tokens: Whether to add special tokens to the input.
            Note that whether BOS or EOS tokens are added depends on the tokenizer.
        amp_dtype: The dtype for automatic mixed precision.
        random_seed: Random seed for the model.
        load_peft: Should be set to True when loading the model from PEFT weights.
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
        amp_dtype: Literal["float16", "bfloat16"] | None = None,
        random_seed: int = 42,
        load_peft: bool = False,
        custom_chat_template: str | None = None,
        default_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_name_or_path = model
        tokenizer = tokenizer if tokenizer else model
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)
        self.custom_chat_template = custom_chat_template
        self.add_special_tokens = add_special_tokens
        self.default_gen_kwargs = default_gen_kwargs or {}

        model_kwargs = model_kwargs or {}
        model_kwargs = {**model_kwargs}  # copy kwargs to avoid modifying the original dict
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = "auto"
        if "torch_dtype" not in model_kwargs:
            # You need to set torch_dtype to use the optimal dtype for the model.
            # https://huggingface.co/docs/transformers/main/main_classes/model#model-instantiation-dtype
            model_kwargs["torch_dtype"] = "auto"
        elif model_kwargs["torch_dtype"] != "auto":
            # Convert string to torch.dtype
            # We allow either "bfloat16" or "torch.bfloat16"
            torch_dtype_str = model_kwargs["torch_dtype"]
            if torch_dtype_str.startswith("torch."):
                torch_dtype_str = torch_dtype_str[len("torch.") :]
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype_str)
            if not isinstance(model_kwargs["torch_dtype"], torch.dtype):
                msg = f"Invalid torch_dtype: {model_kwargs['torch_dtype']}"
                raise ValueError(msg)

        if not load_peft:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model,
                **model_kwargs,
            )
        else:
            from peft import AutoPeftModelForCausalLM

            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model,
                **model_kwargs,
            )

        self.model.eval()

        self.amp_dtype = amp_dtype

        transformers.set_seed(random_seed)

        logger.info(f"model device: {self.model.device}")
        logger.info(f"model dtype: {self.model.dtype}")
        logger.info(f"amp_dtype: {amp_dtype}")
        logger.info(f"random seed: {random_seed}")

    def _get_amp_context(self) -> contextlib.AbstractContextManager:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        if self.amp_dtype == "float16":
            return torch.amp.autocast(
                device_type=self.model.device.type,
                dtype=torch.float16,
            )
        if self.amp_dtype == "bfloat16":
            return torch.amp.autocast(
                device_type=self.model.device.type,
                dtype=torch.bfloat16,
            )

        msg = f"Invalid amp_dtype: {self.amp_dtype}"
        raise ValueError(msg)

    def _get_stop_token_ids(self, stop_sequences: list[str]) -> list[int]:
        stop_token_ids: list[int] = []
        for stop_seq in stop_sequences:
            # Try to convert string to id using `convert_tokens_to_ids`
            # We do not use the `encode` method
            # because in the case of sentencepiece-based tokenizers,
            # calling the encode method adds a redundant space at the beginning of the string,
            stop_token_id = self.tokenizer.convert_tokens_to_ids(stop_seq)

            # NeoXTokenizer returns Unk when calling convert_tokens_ids
            # because each token is stored in a peculiar way
            # Ex. "」" -> "ãĢį"
            if stop_token_id == self.tokenizer.unk_token_id:
                # In such a case, we try to get the ID by calling the encode method.
                stop_seq_tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                if stop_seq_tokens:
                    stop_token_id = stop_seq_tokens[-1]
            # If the token does not match the specified string itself, we do not include it as a stop token id
            if self.tokenizer.decode(stop_token_id) != stop_seq:
                continue

            stop_token_ids.append(stop_token_id)
        return stop_token_ids

    @torch.inference_mode()
    def batch_complete_text(
        self,
        text_list: list[str],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        ignore_eos: bool = False,
        include_stop_str_in_output: bool = False,
        **kwargs,
    ) -> list[str]:
        gen_kwargs = self.default_gen_kwargs.copy()
        gen_kwargs.update(kwargs)
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        model_inputs = tokenize_text_for_lm_prefix(
            text_list,
            self.tokenizer,
            add_special_tokens=self.add_special_tokens,
        ).to(self.model.device)
        input_token_length = model_inputs["input_ids"].shape[1]

        # set the stop sequences
        stop_sequences = normalize_stop_sequences(
            stop_sequences_list=[
                stop_sequences,
                gen_kwargs.pop("stop_strings", None),  # This is used in the transformers `generate` function
                gen_kwargs.pop("stop_sequences", None),  # This is a common variable name used in flexeval
            ],
            eos_token=self.tokenizer.eos_token,
            ignore_eos=ignore_eos,
        )
        stop_token_ids = self._get_stop_token_ids(stop_sequences)
        gen_kwargs.update(
            {
                "eos_token_id": stop_token_ids,
                "pad_token_id": self.tokenizer.pad_token_id,
            },
        )

        with self._get_amp_context():
            lm_outputs = self.model.generate(**model_inputs, **gen_kwargs)

        # `lm_outputs` contains full text including the input text.
        # We strip the input text and stop sequences from the output text.
        output_texts: list[str] = []
        for output_tensor in lm_outputs[:, input_token_length:]:
            output_tokens = [t for t in output_tensor.tolist() if t != self.tokenizer.pad_token_id]
            decoded_text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)

            if include_stop_str_in_output:
                output_texts.append(decoded_text)
                continue

            for stop_seq in stop_sequences:
                idx = decoded_text.find(stop_seq)
                if idx != -1:
                    decoded_text = decoded_text[:idx]
            output_texts.append(decoded_text)
        return output_texts

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

    @torch.inference_mode()
    def batch_compute_log_probs(
        self,
        text_list: list[str],
        prefix_list: list[str] | None = None,
        stride: int | None = None,
    ) -> list[float]:
        batch_size = len(text_list)

        # prepare prefix encoding
        prefix_list = prefix_list if prefix_list else ["" for _ in range(batch_size)]
        # If the prefix is an empty string, replace it with the bos token regardless of the model being trained with it.
        # This is needed to correctly calculate the log probabilities of the first token.
        for i in range(batch_size):
            if prefix_list[i] == "":
                prefix_list[i] = self.tokenizer.bos_token

        prefix_encoding = tokenize_text_for_lm_prefix(
            prefix_list,
            self.tokenizer,
            add_special_tokens=self.add_special_tokens,
        )

        # prepare continuation encoding
        # If the last token is a special token, it is treated as a beginning of a new sentence.
        continuation_encoding = tokenize_text_for_lm_continuation(
            text_list,
            self.tokenizer,
            as_continuation=[
                prefix_ids[-1] not in self.tokenizer.all_special_ids for prefix_ids in prefix_encoding.input_ids
            ],
        )

        input_data_dict: dict[str, torch.Tensor] = {}
        for key in continuation_encoding:
            input_data_dict[key] = torch.cat(
                [prefix_encoding[key].long(), continuation_encoding[key].long()],
                dim=1,
            )
        input_encoding = BatchEncoding(input_data_dict)

        max_length = self.model.config.max_position_embeddings
        stride = stride or max_length // 2
        if not (0 < stride < max_length):
            msg = f"stride must be in (0, {max_length}), but got {stride}"
            raise ValueError(msg)
        sequence_length = input_encoding.input_ids.size(1)

        with self._get_amp_context():
            # stores log probabilities of the next token for each input token
            last_computed_index: int = 0
            log_prob_of_next = torch.zeros_like(
                input_encoding.input_ids,
                dtype=torch.float32,
            )
            for chunk_start in range(0, sequence_length, stride):
                chunk_end = min(chunk_start + max_length, sequence_length)

                # Visualize the input / output processing
                # input_encoding.input_ids: [ 0  1  2  3  4 ]
                # chunk_input_ids:          [ 0  1  2  3    ]
                # chunk_target_ids:         [    1  2  3  4 ]

                input_start = chunk_start
                input_end = chunk_end - 1

                chunk_input_ids = input_encoding.input_ids[:, input_start:input_end].to(self.model.device)
                chunk_input_mask = input_encoding.attention_mask[:, input_start:input_end].to(self.model.device)
                chunk_target_ids = input_encoding.input_ids[:, chunk_start + 1 : chunk_end].to(self.model.device)

                chunkmodel_inputs = self.model.prepare_inputs_for_generation(
                    chunk_input_ids,
                    attention_mask=chunk_input_mask,
                )
                lm_outputs = self.model.forward(**chunkmodel_inputs)

                chunk_log_probs = F.log_softmax(lm_outputs.logits, dim=-1)
                # shape of chunk_log_probs: (batch_size, sequence_length, vocab_size)
                # shape of target_ids: (batch_size, sequence_length)
                # get the log probs of the target ids
                chunk_next_log_probs = chunk_log_probs.gather(
                    dim=-1,
                    index=chunk_target_ids.unsqueeze(-1),
                ).squeeze(-1)

                log_prob_of_next[:, last_computed_index:input_end] = chunk_next_log_probs[
                    :,
                    last_computed_index - input_start :,
                ]

                last_computed_index = input_end

                if chunk_end == sequence_length:
                    break

            log_prob_mask = input_encoding.attention_mask.clone()
            # replace the last token's log prob with 0
            for i in range(log_prob_mask.shape[0]):
                last_non_pad_index = log_prob_mask[i].nonzero(as_tuple=True)[0][-1].item()
                log_prob_mask[i, last_non_pad_index] = 0
            # mask out log probs of prefix tokens
            prefix_length = prefix_encoding.input_ids.shape[1]
            if prefix_length > 0:
                log_prob_mask[:, : prefix_length - 1] = 0
            total_log_probs = (log_prob_of_next * log_prob_mask).sum(dim=-1)
        return total_log_probs.tolist()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model_name_or_path!r})"
