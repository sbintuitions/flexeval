from __future__ import annotations

from typing import Any

import torch


def get_default_model_kwargs(model_kwargs: None | dict[str, Any] = None) -> dict[str, Any]:
    """
    This provides a default set of model kwargs for initializing a model in flexeval.
    - It uses the "auto" device_map to use any available device (GPU, MPS, etc.).
    - It uses the "auto" torch_dtype to use the optimal dtype for the model.
    """
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
    return model_kwargs
