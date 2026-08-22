"""Tests for mcp_bridge: config loading and tool schema conversion."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp_bridge  # noqa: E402


def test_to_openai_schema():
    from types import SimpleNamespace

    tool = SimpleNamespace(name="add", description="Add two ints", input_schema={"type": "object", "properties": {"a": {"type": "integer"}}})
    schema = mcp_bridge.MCPClient._to_openai_schema(tool)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert schema["function"]["parameters"] == tool.input_schema


def test_to_openai_schema_empty():
    from types import SimpleNamespace

    tool = SimpleNamespace(name="t", description=None, input_schema=None)
    schema = mcp_bridge.MCPClient._to_openai_schema(tool)
    assert schema["function"]["description"] == ""
    assert schema["function"]["parameters"]["type"] == "object"


def test_load_config_from_file(tmp_path, monkeypatch):
    cfg = tmp_path / "mcp_config.json"
    cfg.write_text(json.dumps({"servers": [{"name": "s1", "command": ["python", "x.py"]}]}))
    monkeypatch.setattr(mcp_bridge, "MCP_CONFIG_PATH", str(cfg))
    servers = mcp_bridge.load_config()
    assert servers[0]["name"] == "s1"


def test_load_config_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_bridge, "MCP_CONFIG_PATH", str(tmp_path / "nope.json"))
    assert mcp_bridge.load_config() == []


def test_client_constructor_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_bridge, "MCP_CONFIG_PATH", str(tmp_path / "nope.json"))
    c = mcp_bridge.MCPClient()
    assert c.servers == []


def test_standalone_main(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_bridge, "MCP_CONFIG_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setattr(sys, "argv", ["mcp_bridge.py"])
    mcp_bridge.main()
    out = capsys.readouterr().out
    assert "loaded 0 tool(s)" in out
