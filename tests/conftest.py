def is_vllm_enabled() -> bool:
    try:
        import torch
        import vllm  # noqa: F401

        return torch.cuda.device_count() > 0
    except ImportError:
        return False
