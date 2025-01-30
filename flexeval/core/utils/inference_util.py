import re


def separate_reasoning_and_content(
    text: str,
    reasoning_start_tag: str = "<think>",
    reasoning_end_tag: str = "</think>",
) -> dict[str, str]:
    """
    Separates the reasoning part and the content part from the text.

    Args:
        text: The text to be analyzed.
        think_start_tag: The starting tag for the reasoning part.
        think_end_tag: The ending tag for the reasoning part.

    Returns:
        dict: A dictionary containing the separation results.
    """
    start = re.escape(reasoning_start_tag)
    end = re.escape(reasoning_end_tag)
    pattern = re.compile(rf"{start}(.*?){end}(.*)")

    if match := pattern.search(text):
        reasoning = match.group(1).strip() if match.group(1) else ""
        content = match.group(2).strip() if match.group(2) else text.strip()
        return {"reasoning": reasoning, "content": content}

    return {"reasoning": "", "content": text.strip()}
