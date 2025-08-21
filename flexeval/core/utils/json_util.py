from dataclasses import asdict, is_dataclass
from typing import Any


def truncated_default(o: Any, max_length: int = 512) -> Any:  # noqa: ANN401
    if is_dataclass(o):
        return asdict(o)

    s = str(o)
    if len(s) > max_length:
        return s[:max_length] + f"... [truncated, {len(s)} chars total]"
    return s
