from flexeval.core.utils.inference_util import separate_reasoning_and_content


def test_separate_reasoning_and_content() -> None:
    result = separate_reasoning_and_content("<think>ええと</think>それは正しい日本語ですか？")
    assert result["reasoning"] == "ええと"
    assert result["content"] == "それは正しい日本語ですか？"

    result = separate_reasoning_and_content("これは正しい日本語です")
    assert result["reasoning"] == ""
    assert result["content"] == "これは正しい日本語です"
