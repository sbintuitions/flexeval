from __future__ import annotations

import contextlib
import logging
from typing import Any, Literal, TypeVar

import torch
import torch.nn.functional as F  # noqa: N812
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedModel, PreTrainedTokenizer
from .chunkllama.chunkllama_attn_replace import replace_with_chunkllama, replace_with_chunkmistral, replace_with_chunkmixtral
from .chunkllama.chunkqwen_attn_replace import replace_with_chunkqwen

from .hf_lm import HuggingFaceLM

logger = logging.getLogger(__name__)

T = TypeVar("T")

class ChunkLlamaHuggingFaceLM(HuggingFaceLM):

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_name: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        add_special_tokens: bool = False,
        amp_dtype: Literal["float16", "bfloat16"] | None = None,
        random_seed: int = 42,
        load_peft: bool = False,
        custom_chat_template: str | None = None,
        inference_with_dca: bool = False,
        local_window_size_of_dca: int | None = None,
        full_logits_size_of_qwen2_dca: int | None = None,
    ) -> None:
        tokenizer_name = tokenizer_name if tokenizer_name else model_name
        tokenizer_kwargs = tokenizer_kwargs or {}
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        self._custom_chat_template = custom_chat_template
        self._add_special_tokens = add_special_tokens

        model_kwargs = model_kwargs or {}
        model_kwargs = {**model_kwargs}  # copy kwargs to avoid modifying the original dict
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = "auto"
        if "torch_dtype" not in model_kwargs or model_kwargs["torch_dtype"] == "auto":
            # You need to set torch_dtype to use the optimal dtype for the model.
            # https://huggingface.co/docs/transformers/main/main_classes/model#model-instantiation-dtype
            model_kwargs["torch_dtype"] = "auto"
        else:
            # Convert string to torch.dtype
            model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])
            if not isinstance(model_kwargs["torch_dtype"], torch.dtype):
                msg = f"Invalid torch_dtype: {model_kwargs['torch_dtype']}"
                raise ValueError(msg)

        if inference_with_dca:
            config = AutoConfig.from_pretrained(model_name)
            pretraining_length = config.max_position_embeddings
            architecture = config.architectures[0]

            assert architecture in ['LlamaForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM', 'Qwen2ForCausalLM'], \
                f'{architecture} is not supported to inference with dca. Please use a model with Llama, Mistral, Mixtral, or Qwen2 architecture.'
            
            if not architecture == 'Qwen2ForCausalLM':
                if full_logits_size_of_qwen2_dca is not None:
                    logger.warning(f'full_logits_size_of_qwen_dca should be set only when a model with Qwen2ForCausalLM is used. But you use a model with {architecture} and you do not have to set it.')

                if architecture == 'LlamaForCausalLM':
                    replace_with_chunkllama(pretraining_length=pretraining_length, local_window_size=local_window_size_of_dca)
                elif architecture == 'MistralForCausalLM':
                    replace_with_chunkmistral(pretraining_length=pretraining_length, local_window_size=local_window_size_of_dca)
                elif architecture == 'MixtralForCausalLM':
                    replace_with_chunkmixtral(pretraining_length=pretraining_length, local_window_size=local_window_size_of_dca)
            else:
                replace_with_chunkqwen(pretraining_length=pretraining_length, local_window_size=local_window_size_of_dca, full_logits_size=full_logits_size_of_qwen2_dca)
            
            logging.info(f'The model has {architecture} and now it will infrence with dual chunk attention. It can process a longer context than pretrained length {pretraining_length}.')

        else:
            if local_window_size_of_dca is not None:
                logger.warning(f'local_window_size_of_dca is utilized only when inference_with_dca=True. Now, inference_with_dca=False and it\'s ignored.')
            elif full_logits_size_of_qwen2_dca is not None:
                logger.warning(f'full_logits_size_of_qwen2_dca is utilized only when inference_with_dca=True. Now, inference_with_dca=False and it\'s ignored.')

        if not load_peft:
            self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        else:
            from peft import AutoPeftModelForCausalLM

            self._model = AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            # For models such as LoRA, we can merge the additional weights to run inference faster.
            if hasattr(self._model, "merge_and_unload"):
                self._model: PreTrainedModel = self._model.merge_and_unload()

        self._model.eval()

        self._amp_dtype = amp_dtype

        transformers.set_seed(random_seed)

        logger.info(f"model device: {self._model.device}")
        logger.info(f"model dtype: {self._model.dtype}")
        logger.info(f"amp_dtype: {amp_dtype}")
        logger.info(f"random seed: {random_seed}")
