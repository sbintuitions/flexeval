import pytest

from flexeval import LastChoiceExtractor


@pytest.mark.parametrize(
    ("lm_output", "reference"),
    [
        ("The answer is A.", "A"),
        ("A is the answer.", "A"),
        ("My final answer is B.", "B"),
        ("I first thought (A), but after reconsidering I chose (B).", "B"),
        ("The answer is (B).", "B"),
        ("No option is mentioned here.", ""),
    ],
)
def test_multiple_choice_extractor_en(lm_output: str, reference: str) -> None:
    extractor = LastChoiceExtractor(lang="en")
    assert extractor(lm_output) == reference


@pytest.mark.parametrize(
    ("lm_output", "reference"),
    [
        ("答えはAです。", "A"),
        ("最初はAが正解だと思いましたが、最終的にはBが正解です。", "B"),
        ("正解はBです。", "B"),
        ("正解は(B)です。", "B"),
        ("正解は（B）です。", "B"),
        ("わかりました。答えはBですね 。", "B"),
    ],
)
def test_multiple_choice_extractor_ja(lm_output: str, reference: str) -> None:
    extractor = LastChoiceExtractor(lang="ja")
    assert extractor(lm_output) == reference


def test_multiple_choice_extractor_number_options() -> None:
    extractor = LastChoiceExtractor(lang="en", option_type="number", max_num_options=3)
    assert extractor("The answer is 2.") == "2"
    assert extractor("The answer is 5.") == ""


def test_multiple_choice_extractor_lowercase_options() -> None:
    extractor = LastChoiceExtractor(lang="en", option_type="alphabet")
    assert extractor("The answer is (c).") == "c"


def test_multiple_choice_extractor_invalid_option_type() -> None:
    with pytest.raises(ValueError, match="Unsupported option_type"):
        LastChoiceExtractor(option_type="roman")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
