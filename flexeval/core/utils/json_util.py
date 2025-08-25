import re
from dataclasses import asdict, is_dataclass
from typing import Any

base64_pattern = re.compile(r"^data:\w+\/\w+;base64,.+")


def truncate_base64(o: Any) -> Any:  # noqa: ANN401
    if is_dataclass(o):
        return asdict(o)

    s = str(o)

    if base64_pattern.match(s):
        return f"{s[:50]}... [truncated, {len(s)} chars total]"

    return s
