from __future__ import annotations

import platform
from typing import Any

import torch


def is_pre_macos_14() -> bool:
    """
    Returns True if the macOS version is older than 14.0

    This function is useful for avoiding MPS-related issues on older macOS versions,
        such as the following error:
    "RuntimeError: isin_Tensor_Tensor_out only works on floating types on MPS for pre MacOS_14_0"
    """
    if platform.system() != "Darwin":
        return False  # Not running on macOS

    version = platform.mac_ver()[0]
    if not version:
        return False  # Could not determine macOS version

    major_version = int(version.split(".")[0])
    return major_version < 14


def get_default_model_kwargs(model_kwargs: None | dict[str, Any] = None) -> dict[str, Any]:
    """
    This provides a default set of model kwargs for initializing a model in flexeval.
    - It uses the "auto" device_map to use any available device (GPU, MPS, etc.).
    - It uses the "auto" dtype to use the optimal dtype for the model.
    """
    model_kwargs = model_kwargs or {}
    model_kwargs = {**model_kwargs}  # copy kwargs to avoid modifying the original dict
    if "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "auto"
        if is_pre_macos_14():
            # If the macOS version is older than 14.0, we need to use "cpu" instead of "auto".
            # This is to avoid MPS-related issues on older macOS versions.
            model_kwargs["device_map"] = "cpu"
    # `torch_dtype` was renamed to `dtype` in transformers v5. Keep accepting the old
    # name so existing configs don't break.
    if "torch_dtype" in model_kwargs:
        model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
    if "dtype" not in model_kwargs:
        # You need to set dtype to use the optimal dtype for the model.
        # https://huggingface.co/docs/transformers/main/main_classes/model#model-instantiation-dtype
        model_kwargs["dtype"] = "auto"
    elif model_kwargs["dtype"] != "auto":
        # Convert string to torch.dtype
        # We allow either "bfloat16" or "torch.bfloat16"
        dtype_str = model_kwargs["dtype"]
        dtype_str = dtype_str.removeprefix("torch.")
        model_kwargs["dtype"] = getattr(torch, dtype_str)
        if not isinstance(model_kwargs["dtype"], torch.dtype):
            msg = f"Invalid dtype: {model_kwargs['dtype']}"
            raise ValueError(msg)
    return model_kwargs
