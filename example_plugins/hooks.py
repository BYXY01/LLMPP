"""Example plugin: inbound/outbound hooks.

Copy to `plugins/` to use, then enable in config.json:
    "hooks": {"inbound": "inbound", "outbound": "outbound"}
"""


def inbound(messages):
    """Inbound hook: runs before messages reach the LLM.

    Args:
        messages: The request messages list.

    Returns:
        The processed messages list.
    """
    print(f"[inbound] received {len(messages)} messages")
    return messages


def outbound(messages, stream_chunk=None):
    """Outbound hook: runs after the LLM replies, before returning.

    Args:
        messages: The reply messages list.
        stream_chunk: During streaming, the current text chunk.

    Returns:
        The processed messages list (or (messages, chunk) while streaming).
    """
    print(f"[outbound] reply messages: {len(messages)}")
    if stream_chunk is not None:
        return messages, stream_chunk
    return messages


__hooks__ = [inbound, outbound]
