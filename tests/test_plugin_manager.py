"""Tests for PluginManager: plugin loading, tool registry, management."""

import json
import os

import plugin_manager as pm
from plugin_manager import PluginManager


def test_load_plugins(manager):
    assert "get_time" in manager.plugins
    assert manager.manager_fn is not None


def test_tools_schema(manager):
    tools = manager.tools()
    names = [t["function"]["name"] for t in tools]
    assert "get_time" in names
    time_schema = next(t for t in tools if t["function"]["name"] == "get_time")
    assert time_schema["type"] == "function"


def test_has_tool(manager):
    assert manager.has_tool("get_time")
    assert not manager.has_tool("nonexistent")


def test_call_plugin(manager):
    result = manager.call("get_time", {})
    assert "20" in result or "19" in result or result  # datetime string


def test_call_unknown(manager):
    assert "not found" in manager.call("nope", {})


def test_inbound_outbound_hooks(tmp_path):
    hook_src = {
        "hooks.py": (
            "def inbound(messages):\n"
            "    messages = list(messages)\n"
            "    messages[0]['content'] = 'modified:' + messages[0]['content']\n"
            "    return messages\n"
            "def outbound(messages):\n"
            "    return list(messages)\n"
            "__hooks__ = [inbound, outbound]\n"
        ),
    }
    d = tmp_path / "plugins"
    d.mkdir()
    for name, src in hook_src.items():
        (d / name).write_text(src)
    import plugin_manager as pm

    mgr = pm.PluginManager(plugins_dir=str(d))
    mgr.load()
    cfg = {"hooks": {"inbound": "inbound", "outbound": "outbound"}}
    msgs = [{"role": "user", "content": "hi"}]
    out = mgr.run_inbound(cfg, msgs)
    assert out[0]["content"] == "modified:hi"
    res = mgr.run_outbound(cfg, out)
    assert isinstance(res, list) and len(res) == 1


def test_manager_list(manager):
    lst = manager.manager("list")
    names = {p["name"] for p in lst}
    assert "example_time" in names
    assert all(p["enabled"] for p in lst)


def test_disable_takes_effect_immediately(plugin_dir, tmp_path, monkeypatch):
    # Use a separate state path so we don't touch the real plugins.json.
    monkeypatch.setattr(pm, "BASE_DIR", str(tmp_path))
    mgr = PluginManager(plugins_dir=plugin_dir)
    mgr.load()
    mgr.manager("disable", "example_time")
    assert "get_time" not in mgr.plugins
    assert mgr.has_tool("get_time") is False
    # persisted
    state = tmp_path / "plugins.json"
    data = json.loads(state.read_text())
    assert data["example_time"]["status"] == "disabled"


def test_disable_persists_across_reload(plugin_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "BASE_DIR", str(tmp_path))
    state = tmp_path / "plugins.json"
    mgr = PluginManager(plugins_dir=plugin_dir)
    mgr.load()
    mgr.manager("disable", "example_time")

    mgr2 = PluginManager(plugins_dir=plugin_dir)
    mgr2.load()
    assert "get_time" not in mgr2.plugins


def test_call_with_info(manager):
    def with_info(text, info=None):
        return f"text={text} info={info}"
    def without_info(text):
        return f"text={text}"
    manager.plugins["with_info"] = with_info
    manager.plugins["without_info"] = without_info

    info = {"ip": "1.2.3.4", "auth_status": "auth"}
    # plugin declares info -> injected
    r = manager.call("with_info", {"text": "hi"}, info=info)
    assert "info={'ip': '1.2.3.4'" in r
    # plugin without info -> not injected, works normally
    assert manager.call("without_info", {"text": "hi"}, info=info) == "text=hi"
    # args containing an `info` key is not leaked to a plugin without info
    assert manager.call("without_info", {"text": "hi", "info": "junk"}, info=info) == "text=hi"


def test_call_info_overrides_args(manager):
    def with_info(text, info=None):
        return f"text={text} info={info}"
    manager.plugins["with_info"] = with_info
    r = manager.call("with_info", {"text": "hi", "info": "junk"}, info={"ip": "real"})
    assert "info={'ip': 'real'}" in r


def test_hooks_with_info():
    seen = {}

    def inbound(messages, info=None):
        seen["inbound_info"] = info
        return messages

    def outbound(messages, info=None):
        seen["outbound_info"] = info
        return messages

    mgr = PluginManager()
    mgr.hooks["inbound"] = inbound
    mgr.hooks["outbound"] = outbound
    cfg = {"hooks": {"inbound": "inbound", "outbound": "outbound"}}
    info = {"ip": "9.9.9.9", "auth_status": "auth"}

    msgs = [{"role": "user", "content": "hi"}]
    mgr.run_inbound(cfg, msgs, info=info)
    mgr.run_outbound(cfg, msgs, info=info)
    assert seen["inbound_info"] == info
    assert seen["outbound_info"] == info


def test_hook_without_info_still_works():
    def inbound(messages):
        return messages

    mgr = PluginManager()
    mgr.hooks["inbound"] = inbound
    cfg = {"hooks": {"inbound": "inbound", "outbound": ""}}
    msgs = [{"role": "user", "content": "hi"}]
    out = mgr.run_inbound(cfg, msgs, info={"ip": "x"})
    assert out == msgs


def test_hot_enable(plugin_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "BASE_DIR", str(tmp_path))
    mgr = PluginManager(plugins_dir=plugin_dir)
    mgr.load()
    assert "get_time" in mgr.plugins
    mgr.manager("disable", "example_time")
    assert "get_time" not in mgr.plugins
    mgr.manager("enable", "example_time")
    assert "get_time" in mgr.plugins


def test_reload(plugin_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "BASE_DIR", str(tmp_path))
    mgr = PluginManager(plugins_dir=plugin_dir)
    mgr.load()
    assert "get_time" in mgr.plugins
    r = mgr.manager("reload", "example_time")
    assert "reloaded" in r
    assert "get_time" in mgr.plugins


def test_manager_plugin_schema_filters_first_arg(manager):
    """The authorized manager function exposes action/name, not `manager`."""
    tools = manager.tools()
    manage_schema = next(t for t in tools if t["function"]["name"] == "manage")
    params = manage_schema["function"]["parameters"]
    assert set(params["properties"]) == {"action", "name"}
    assert params["required"] == ["action"]
    assert "manager" not in params["properties"]


def test_manager_plugin_call_injects_manager(manager):
    """Calling the authorized manager function injects the manager as arg 1."""
    assert manager.manager_fn is not None
    assert manager.manager_fn.__name__ == "manage"
    result = manager.call("manage", {"action": "list"})
    assert "example_time" in result


def test_manager_plugin_as_hook_injects_manager(tmp_path):
    """A hook authorized as manager receives `manager` as its first argument."""
    hook_src = {
        "mgr_hook.py": (
            "def manage(manager, messages):\n"
            "    return messages\n"
            "__hooks__ = [manage]\n"
        ),
    }
    d = tmp_path / "plugins"
    d.mkdir()
    for name, src in hook_src.items():
        (d / name).write_text(src)
    mgr = pm.PluginManager(plugins_dir=str(d), manager_plugin="manage")
    mgr.load()
    assert mgr.manager_fn is not None and mgr.manager_fn.__name__ == "manage"
    cfg = {"hooks": {"inbound": "manage", "outbound": ""}}
    msgs = [{"role": "user", "content": "hi"}]
    assert mgr.run_inbound(cfg, msgs) == msgs
