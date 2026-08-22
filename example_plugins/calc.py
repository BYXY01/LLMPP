"""Example plugin: basic calculator with typed parameters.

Copy to `plugins/` to use, or point PluginManager at this directory.
"""


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of a and b.
    """
    return a * b


__tools__ = [add, multiply]
