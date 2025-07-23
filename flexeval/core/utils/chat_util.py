from __future__ import annotations


def find_first_turn_for_response(messages: list[dict]) -> int:
    """
    Returns the index of the first message in the conversation that should be responded to,
    skipping messages from system or developer roles.

    Args:
        messages (list[dict]): List of messages, each as a dict with at least a "role" key.

    Returns:
        int: The index of the first message that should be responded to.
    """
    for i, m in enumerate(messages):
        if m["role"] not in {"system", "developer"}:
            return i
    return len(messages)
