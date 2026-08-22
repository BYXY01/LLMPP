# LLMPP - LLM Plugin Proxy

```
   ______         __    __    __  _______  ____ 
  /     /|       / /   / /   /  |/  / __ \/ __ \
 /_____/ |      / /   / /   / /|_/ / /_/ / /_/ /
 |     | |     / /___/ /___/ /  / / ____/ ____/ 
 |_____|/     /_____/_____/_/  /_/_/   /_/      
```

An out-of-the-box, **OpenAI-compatible** middleware that extends any LLM backend with plugin capabilities.

```
Caller (OpenAI / Anthropic)
    ↓ inbound conversion (unified internal OpenAI)
LLM_Server (async pipeline, waitress thread)
    ├─ PluginManager (plugin management thread)
    └─ MCPClient (optional MCP peer, own thread)
    ↓ outbound conversion (llm.provider format)
Backend (OpenAI / LM Studio / Anthropic / ...)
```

## Features

- **Zero-setup run** — auto-installs missing core deps on boot.
- **OpenAI-compatible API** — `POST /v1/chat/completions`, works with any OpenAI SDK/client.
- **Drop-in plugins** — put `.py` files in `plugins/`, they auto-load on startup.
- **Two protocols**:
  - `native` — native OpenAI function/tool calling.
  - `compatible` — fallback for models without tools support, with a two-tier ladder (structured JSON schema, then prompt + fault-tolerant parsing).
- **Hooks** — inbound/outbound message processors (1 each, chosen in config). Outbound supports optional streaming chunks.
- **Plugin deps** — a plugin can declare `__deps__` and LLMPP auto-installs them.
- **Streaming** (experimental) — native mode supports SSE streaming (token-by-token), gated by `server.stream`.
- **Async & threaded** — async request pipeline (AsyncOpenAI); PluginManager management runs on its own thread, sharing a locked registry.
- **Dual protocols** — OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` inbound; `llm.provider` selects OpenAI or Anthropic backend.
- **Full `/v1/*` passthrough** — optional `routes.full_v1` proxies models/embeddings/etc. to the backend.
- **Modular files** — `plugin_manager.py`, `llm_server.py`, and the `LLMPP.py` entry split the two main classes.
- **Caller tools** — client-provided `tools` are merged; caller-owned tools are passed back to the client to execute (standard agentic loop).
- **Auth & key passthrough** — optional API keys via the `LLMPP_API_KEYs` env var; caller-provided keys can be passed through to the provider.
- **Production WSGI** — served by waitress, not the Flask development server.
- **MCP client** (experimental) — connect external MCP servers (stdio/HTTP) via the standalone `mcp_bridge.py`; their tools join the tool registry.
- **Plugin management** — an authorized manager function (via `manager_plugin`) can list/enable/disable/reload plugins; state persists in `plugins.json`.
- **Any LLM backend** — OpenAI, LM Studio, Ollama, vLLM, etc. (any OpenAI-compatible base URL).

## Quick Start

```bash
git clone https://github.com/BYXY01/LLMPP.git
cd LLMPP
python LLMPP.py
```

First run generates `config.json`, then exits. Edit it, then run again:

```bash
python LLMPP.py
```

## Configuration

`config.json` (auto-generated):

```json
{
    "server": {
        "host": "127.0.0.1",
        "port": 55677,
        "stream": false,
        "api_keys": []
    },
    "llm": {
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "timeout": 120
    },
    "mode": "native",
    "tools": {
        "max_rounds": 10
    },
    "hooks": {
        "inbound": "",
        "outbound": ""
    }
}
```

| Field | Description |
|-------|-------------|
| `server.host` / `server.port` | Bind address and port. `port` is required. |
| `server.stream` | Enable SSE streaming (experimental, native mode only). |
| `server.api_keys` | LLMPP auth keys (max 5); may include `_PASSTHROUGH_API_KEY`. See "Auth & key passthrough". |
| `llm.api_base` / `api_key` | Your OpenAI-compatible backend (LM Studio, Ollama, vLLM, ...). |
| `llm.provider` | Backend format: `openai` (default) or `anthropic` (Claude). |
| `mode` | `native` (function calling) or `compatible` (text protocol). |
| `routes.full_v1` | Proxy the whole `/v1/*` namespace (models, embeddings, ...) to the backend. |
| `tools.max_rounds` | Max tool-call rounds per request (safety valve against loops). |
| `hooks.inbound` / `hooks.outbound` | Name of the inbound/outbound hook plugin to use. |
| `manager_plugin` | Name of the single authorized manager function (list/enable/disable/reload). |

The model name is **not** configured here — clients pass it in each request, as with the standard OpenAI API.

## Auth & key passthrough

Configure API keys (up to **5**) via the `server.api_keys` list in `config.json`, or the `LLMPP_API_KEYs` environment variable (comma-separated) as a fallback when the config list is empty:

```json
"server": {
    "api_keys": ["my-key-1", "my-key-2"]
}
```

```bash
LLMPP_API_KEYs="my-key-1,my-key-2" python LLMPP.py
```

- **Unset / empty** — LLMPP auth is disabled; requests always call the provider with `llm.api_key`.
- **Strict mode** (no sentinel) — clients must send `Authorization: Bearer <one-of-the-keys>`. A hit calls the provider with `llm.api_key`; a miss returns **401**.
- **Passthrough mode** — include the sentinel `_PASSTHROUGH_API_KEY` in the list. A caller key that hits an LLMPP key is rewritten to `llm.api_key`; a key that doesn't is **passed through to the provider as-is** (so callers can use their own provider keys). Requests without a key get an empty-key passthrough.

```json
"server": {
    "api_keys": ["my-key-1", "_PASSTHROUGH_API_KEY"]
}
```

```bash
LLMPP_API_KEYs="my-key-1,_PASSTHROUGH_API_KEY" python LLMPP.py
```

## MCP client (experimental)

LLMPP can connect to external **MCP servers** (stdio or streamable HTTP) and use their tools alongside plugins. This is **experimental** and entirely optional — if the `mcp` package (or `mcp_bridge.py`) is missing, LLMPP runs normally without MCP.

MCP servers are configured in a **separate** `mcp_config.json` (same directory, not in `config.json`):

```json
{
    "servers": [
        {"name": "math", "command": ["python", "/path/to/mcp_server.py"]},
        {"name": "remote", "url": "http://127.0.0.1:8000/mcp"}
    ]
}
```

`command` entries use the stdio transport; `url` entries use HTTP. Install the SDK to enable:

```bash
python -m pip install mcp
```

List connected tools standalone:

```bash
python mcp_bridge.py
```

MCP tool names join the same tool registry as plugins — the model can call them exactly like plugin tools.

## Plugins

Drop `.py` files into `plugins/`. Plugins need **no imports** — just define functions and declare them. Ready-made examples live in [`example_plugins/`](example_plugins/).

### Tools

```python
# plugins/my_tools.py
def get_time() -> str:
    """Get the current time.

    Returns:
        The current datetime string.
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    """
    return a + b


__tools__ = [get_time, add]
```

The docstring becomes the tool description the model sees. Python type hints drive the JSON Schema types.

### Hooks

```python
# plugins/my_hooks.py
def inbound(messages):
    """Runs before messages reach the LLM. May rewrite the list."""
    return messages


def outbound(messages, stream_chunk=None):
    """Runs on the LLM reply before returning.

    Receives the full reply message list. During streaming, `stream_chunk`
    carries the current text chunk and the hook should return
    (processed_messages, processed_chunk); otherwise it returns the list.
    Hooks without a stream_chunk parameter are skipped during streaming.
    """
    return messages


__hooks__ = [inbound, outbound]
```

Select them in config: `"hooks": {"inbound": "inbound", "outbound": "outbound"}`.

### Plugin dependencies

Declare deps at the top of the file; LLMPP auto-installs them if missing:

```python
__deps__ = ["python-dotenv"]

import os
from dotenv import load_dotenv
```

### Manager plugin

Authorize a **single management function** (by name) in `config.json`:

```json
"manager_plugin": "manage"
```

The authorized function receives `manager` (the plugin manager's single
entry point) as its **first argument**, injected by LLMPP. That argument is
filtered out of the tool schema the model sees, so the model calls it with
just `action`/`name`:

```python
def manage(manager, action, name=None):
    """List/enable/disable/reload plugins."""
    return manager(action, name or "")


__tools__ = [manage]
```

Any function — declared via `__tools__` or `__hooks__` — can be the manager
function; whichever matches `manager_plugin` gets `manager` injected as its
first argument (and, for tools, hidden from the model's schema).

## Usage (API)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:55677/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "What time is it?"}],
)
print(resp.choices[0].message.content)
```

## How it works

- `native` mode: LLMPP merges plugin schemas (and any client-provided `tools`) and sends them as `tools`. Tools owned by LLMPP plugins are executed internally; caller-owned tools are passed back to the client to execute (standard agentic loop).
- `compatible` mode: a two-tier fallback — first structured JSON schema output (no prompt injection), then prompt injection with fault-tolerant JSON extraction. Tool results are fed back as `tool_result:...` / JSON messages.
- Responses include the **full message list** (`messages`), so clients can replace their history and continue the conversation without losing context.
- **Streaming** (native only): text is forwarded token-by-token while tool calls are accumulated and executed internally.

## Status

**Stable** (`v0.1.0`). Core features, auth & key passthrough, plugin management, experimental MCP client; async pipeline + threaded architecture, dual OpenAI/Anthropic protocols & backends, full `/v1/*` passthrough. Automated tests in `tests/` (run `pytest tests/`).

## License

[MIT](LICENSE) © 2026 BYXY01 (XY001)
