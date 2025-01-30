from flexeval.core.utils.inference_util import separate_reasoning_and_content


def test_separate_reasoning_and_content() -> None:
    result = separate_reasoning_and_content("<think>ええと</think>それは正しい日本語ですか？")
    assert result["reasoning"] == "ええと"
    assert result["content"] == "それは正しい日本語ですか？"

    result = separate_reasoning_and_content("これは正しい日本語です")
    assert results[1]["reasoning"] == ""
    assert results[1]["content"] == "これは正しい日本語です"
