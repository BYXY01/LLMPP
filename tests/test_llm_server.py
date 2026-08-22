"""Tests for LLM_Server: routing, auth, Anthropic/OpenAI conversions."""

import asyncio
import json

import pytest

from llm_server import LLM_Server


def make_server(monkeypatch, tmp_path, provider="openai", full_v1=False, api_keys=None):
    import plugin_manager as pm

    monkeypatch.setattr(pm, "BASE_DIR", str(tmp_path))
    mgr = pm.PluginManager(plugins_dir=str(tmp_path / "plugins"))
    mgr.load()
    cfg = {
        "server": {"host": "127.0.0.1", "port": 55677, "stream": False, "api_keys": api_keys or []},
        "llm": {"api_base": "http://127.0.0.1:9999/v1", "api_key": "test", "timeout": 30, "provider": provider},
        "mode": "native",
        "routes": {"full_v1": full_v1},
        "tools": {"max_rounds": 10},
        "hooks": {"inbound": "", "outbound": ""},
        "manager_plugin": "",
    }
    server = LLM_Server(cfg, mgr)
    return server


# --- Anthropic request -> OpenAI conversion ---

def test_from_anthropic_request():
    srv = object.__new__(LLM_Server)
    payload = {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "tu1", "name": "add", "input": {"a": 1, "b": 2}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "3"}]},
        ],
        "tools": [{"name": "add", "description": "add", "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}],
    }
    out = srv._from_anthropic_request(payload)
    assert out["model"] == "claude-x"
    # assistant tool_use -> OpenAI tool_calls
    assert out["messages"][1]["tool_calls"][0]["function"]["name"] == "add"
    assert out["messages"][2]["role"] == "tool"
    # tools converted
    assert out["tools"][0]["function"]["name"] == "add"


# --- OpenAI messages -> Anthropic (backend) conversion ---

def test_to_anthropic_messages():
    srv = object.__new__(LLM_Server)
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "think", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "3"},
    ]
    out = srv._to_anthropic_messages(msgs)
    assert out[0]["role"] == "user"
    # assistant with tool_use block
    blocks = out[1]["content"]
    assert any(b["type"] == "tool_use" and b["name"] == "add" for b in blocks)
    # tool result -> user with tool_result
    assert out[2]["content"][0]["type"] == "tool_result"


def test_to_anthropic_tools():
    srv = object.__new__(LLM_Server)
    tools = [{"type": "function", "function": {"name": "add", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    out = srv._to_anthropic_tools(tools)
    assert out[0]["name"] == "add"
    assert out[0]["input_schema"]["type"] == "object"


def test_from_anthropic_message():
    srv = object.__new__(LLM_Server)

    class Block:
        def __init__(self, t, text=None, tid=None, name=None, inp=None):
            self.type = t
            self.text = text
            self.id = tid
            self.name = name
            self.input = inp

    class Msg:
        content = [
            Block("text", text="res"),
            Block("tool_use", tid="tu", name="add", inp={"a": 5, "b": 6}),
        ]

    msg = srv._from_anthropic_message(Msg())
    assert msg.content == "res"
    assert msg.tool_calls[0]["function"]["name"] == "add"


# --- tool call helper compatibility ---

def test_tc_helpers():
    from llm_server import _tc_name, _tc_args, _tc_id

    obj_tc = type("T", (), {"id": "x", "function": type("F", (), {"name": "get_time", "arguments": "{}"})()})()
    assert _tc_name(obj_tc) == "get_time"
    assert _tc_id(obj_tc) == "x"

    dict_tc = {"id": "y", "function": {"name": "add", "arguments": '{"a":1}'}}
    assert _tc_name(dict_tc) == "add"
    assert _tc_args(dict_tc) == '{"a":1}'


# --- auth ---

def test_resolve_backend_key_strict(monkeypatch, tmp_path):
    srv = make_server(monkeypatch, tmp_path, api_keys=["k1"])
    from flask import Flask

    app = Flask(__name__)
    with app.test_request_context(headers={"Authorization": "Bearer k1"}):
        key, err = srv._resolve_backend_key()
        assert err is None
    with app.test_request_context(headers={"Authorization": "Bearer wrong"}):
        key, err = srv._resolve_backend_key()
        assert err is not None
    with app.test_request_context():
        key, err = srv._resolve_backend_key()
        assert err is not None


def test_resolve_backend_key_noauth(monkeypatch, tmp_path):
    srv = make_server(monkeypatch, tmp_path, api_keys=[])
    from flask import Flask

    app = Flask(__name__)
    with app.test_request_context():
        key, err = srv._resolve_backend_key()
        assert key is None and err is None


# --- routes ---

def test_index_route(monkeypatch, tmp_path):
    srv = make_server(monkeypatch, tmp_path)
    client = srv.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.get_json()["service"] == "LLMPP"


def test_full_v1_registered(monkeypatch, tmp_path):
    srv = make_server(monkeypatch, tmp_path, full_v1=True)
    paths = {r.rule for r in srv.app.url_map.iter_rules()}
    assert "/v1/<path:path>" in paths


def test_full_v1_disabled(monkeypatch, tmp_path):
    srv = make_server(monkeypatch, tmp_path, full_v1=False)
    paths = {r.rule for r in srv.app.url_map.iter_rules()}
    assert "/v1/<path:path>" not in paths
