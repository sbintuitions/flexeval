import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

base64_pattern = re.compile(r"^data:\w+\/\w+;base64,.+")


def _truncate_base64(o: Any) -> Any:  # noqa: ANN401
    if is_dataclass(o):
        return _truncate_base64(asdict(o))
    if isinstance(o, (list, tuple, set)):
        return type(o)(_truncate_base64(item) for item in o)
    if isinstance(o, dict):
        return {k: _truncate_base64(v) for k, v in o.items()}

    s = str(o)

    if base64_pattern.match(s):
        return f"{s[:50]}... [truncated, {len(s)} chars total]"

    return s


class Base64TruncatingJSONEncoder(json.JSONEncoder):
    def encode(self, o: Any) -> str:  # noqa: ANN401
        o = _truncate_base64(o)
        return super().encode(o)
