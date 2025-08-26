import dataclasses
import json

from flexeval.core.utils.json_util import Base64TruncatingJSONEncoder


@dataclasses.dataclass
class TestDataClass:
    field1: str
    field2: int


class TestData:
    def __repr__(self) -> str:
        return "TestData"


base64_string = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgA/"
    "AAAGACAIAAABUQk3oAAEAAElEQVR4nMz9SZgtSXYeiJ1zzMynO0bEizfmPFRW1pAFoFATpmIDLTb"
    "VX5P8WuPXG26p75O2WqillTbSQtroU6sXWojaSE2qG+"
)


def test_truncate_base64() -> None:
    def _json_dumps(x):  # noqa: ANN001, ANN202
        return json.dumps(x, cls=Base64TruncatingJSONEncoder)

    assert _json_dumps(TestDataClass("example", 123)) == {"field1": "example", "field2": 123}

    assert _json_dumps(TestData()) == "TestData"

    assert _json_dumps({"key": base64_string}) == {
        "key": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgA/... [truncated, 169 chars total]"
    }

    assert _json_dumps([base64_string, "normal string"]) == [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgA/... [truncated, 169 chars total]",
        "normal string",
    ]

    assert _json_dumps(TestDataClass(base64_string, 456)) == {
        "field1": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgA/... [truncated, 169 chars total]",
        "field2": 456,
    }

    image_url = _json_dumps({"messages": [{"content": {"type": "image_url", "image_url": {"url": base64_string}}}]})
    assert (
        image_url["messages"][0]["content"]["image_url"]["url"]
        == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgA/... [truncated, 169 chars total]"
    )
