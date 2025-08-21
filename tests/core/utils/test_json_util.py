import dataclasses

from flexeval.core.utils.json_util import truncate_base64


@dataclasses.dataclass
class TestDataClass:
    field1: str
    field2: int


class TestData:
    def __repr__(self) -> str:
        return "TestData"


test_base64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgA/"
    "AAAGACAIAAABUQk3oAAEAAElEQVR4nMz9SZgtSXYeiJ1zzMynO0bEizfmPFRW1pAFoFATpmIDLTb"
    "VX5P8WuPXG26p75O2WqillTbSQtroU6sXWojaSE2qG+"
)


def test_truncate_base64() -> None:
    assert truncate_base64(TestDataClass("example", 123)) == {"field1": "example", "field2": 123}

    assert truncate_base64(TestData()) == "TestData"

    assert (
        truncate_base64(test_base64)
        == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgA/... [truncated, 169 chars total]"
    )

    assert (
        truncate_base64(f"This is an example of base64: {test_base64}")
        == f"This is an example of base64: {test_base64}"
    )
