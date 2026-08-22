"""Shared pytest fixtures for LLMPP tests."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable (LLMPP.py, plugin_manager, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plugin_manager as pm  # noqa: E402

PLUGIN_SRC = {
    "example_time.py": (
        "def get_time() -> str:\n"
        '    """Get the current time."""\n'
        "    from datetime import datetime\n"
        "    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n"
        "__tools__ = [get_time]\n"
    ),
    "example_manager.py": (
        "def manage(manager, action, name=None):\n"
        '    """Manage plugins."""\n'
        "    return manager(action, name or '')\n"
        "__tools__ = [manage]\n"
    ),
}


@pytest.fixture
def plugin_dir(tmp_path):
    d = tmp_path / "plugins"
    d.mkdir()
    for name, src in PLUGIN_SRC.items():
        (d / name).write_text(src)
    return str(d)


@pytest.fixture
def manager(plugin_dir):
    mgr = pm.PluginManager(plugins_dir=plugin_dir, manager_plugin="manage")
    mgr.load()
    return mgr
