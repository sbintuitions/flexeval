import base64
import io
from pathlib import Path

import datasets
import pytest
from PIL import Image
from pytest_mock import MockerFixture

from flexeval import (
    ConvertImageListToBase64,
    ConvertImageToBase64,
    EnsureMinSize,
    HFChatDataset,
)
from flexeval.core.chat_dataset.image_preprocessors import image_to_base64, load_image
from flexeval.core.language_model.base import LMOutput
from tests.dummy_modules import DummyVLM


def _base64_decode(data_url: str) -> Image.Image:
    _, base64_str = data_url.split(",")
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (80, 100), color=(73, 109, 137))


def test_image_to_base64(image: Image.Image) -> None:
    data_url = image_to_base64(image, image_format="PNG")
    assert data_url.startswith("data:image/png;base64,")
    decoded = _base64_decode(data_url)
    assert decoded.size == (80, 100)


def test_image_to_base64_max_length(image: Image.Image) -> None:
    data_url = image_to_base64(image, image_format="JPEG", max_length=850)
    assert data_url.startswith("data:image/jpeg;base64,")
    assert len(data_url.split(",")[1]) <= 850
    decoded = _base64_decode(data_url)
    assert decoded.width < 80


def test_load_image(image: Image.Image, tmp_path: Path) -> None:
    assert load_image(image) is image

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    assert load_image({"bytes": buffer.getvalue(), "path": None}).size == (80, 100)

    image_path = tmp_path / "image.png"
    image.save(image_path)
    assert load_image({"bytes": None, "path": str(image_path)}).size == (80, 100)

    # Intentionally pass a wrong-typed value to exercise the runtime TypeError
    # guard. The signature stays `Image.Image | dict` (not `Any`), so the static
    # checker must be suppressed for this one deliberate violation.
    with pytest.raises(TypeError):
        load_image("not-an-image")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_convert_image_to_base64(image: Image.Image) -> None:
    data = ConvertImageToBase64("image")({"image": image})
    decoded = _base64_decode(data["image_base64"])
    assert decoded.size == (80, 100)


def test_convert_image_to_base64_none() -> None:
    data = ConvertImageToBase64("image")({"image": None})
    assert data["image_base64"] is None


def test_convert_image_list_to_base64(image: Image.Image) -> None:
    data = ConvertImageListToBase64("images")({"images": [image, image]})
    assert len(data["images_base64"]) == 2
    for data_url in data["images_base64"]:
        assert _base64_decode(data_url).size == (80, 100)

    assert ConvertImageListToBase64("images")({"images": None})["images_base64"] is None

    with pytest.raises(TypeError):
        ConvertImageListToBase64("images")({"images": image})


def test_ensure_min_size(image: Image.Image) -> None:
    data = EnsureMinSize("image", min_size=120)({"image": image})
    assert data["image"].size == (120, 150)

    # already large enough: unchanged
    data = EnsureMinSize("image", min_size=50)({"image": image})
    assert data["image"] is image

    assert EnsureMinSize("image", min_size=120)({"image": None})["image"] is None

    with pytest.raises(TypeError):
        EnsureMinSize("image", min_size=120)({"image": "not-an-image"})


def test_hf_chat_dataset_with_image_preprocessors(mocker: MockerFixture, image: Image.Image) -> None:
    mocker.patch(
        "datasets.load_dataset",
        return_value=datasets.Dataset.from_dict(
            {
                "text": ["hello", "bonjour"],
                "image": [image, image],
            },
        ),
    )
    dataset = HFChatDataset(
        path="dummy_path",
        split="train",
        input_template=(
            '[{ "type": "text", "text": """{{ text }}"""},'
            ' { "type": "image_url", "image_url": {"url": "{{ image_base64 }}"}},]'
        ),
        parse_input_utterance="literal_eval",
        preprocessors=[EnsureMinSize("image", min_size=120), ConvertImageToBase64("image")],
    )

    assert len(dataset) == 2
    content = dataset[0].messages[0]["content"]
    assert content[0]["text"] == "hello"
    decoded = _base64_decode(content[1]["image_url"]["url"])
    assert decoded.size == (120, 150)


def test_dummy_vlm_receives_preprocessed_images(mocker: MockerFixture, image: Image.Image) -> None:
    mocker.patch(
        "datasets.load_dataset",
        return_value=datasets.Dataset.from_dict(
            {
                "text": ["hello"],
                "image": [image],
            },
        ),
    )
    dataset = HFChatDataset(
        path="dummy_path",
        split="train",
        input_template=(
            '[{ "type": "text", "text": """{{ text }}"""},'
            ' { "type": "image_url", "image_url": {"url": "{{ image_base64 }}"}},]'
        ),
        parse_input_utterance="literal_eval",
        preprocessors=[EnsureMinSize("image", min_size=120), ConvertImageToBase64("image")],
    )
    lm_output = DummyVLM().generate_chat_response(dataset[0].messages)
    assert isinstance(lm_output, LMOutput)
    assert lm_output.text == "1 image(s) and 1 text part(s) are given!"
