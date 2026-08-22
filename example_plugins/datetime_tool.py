"""Example plugin: current time.

Copy to `plugins/` to use, or point PluginManager at this directory.
"""


def get_time() -> str:
    """Get the current time.

    Returns:
        The current datetime string.
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


__tools__ = [get_time]
